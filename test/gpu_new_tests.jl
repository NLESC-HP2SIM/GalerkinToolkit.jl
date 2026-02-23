module GPUNewTests

using Test
using LinearAlgebra
using SparseArrays
import Adapt
using BenchmarkTools
import PartitionedArrays as PA
import GalerkinToolkit as GT
using KernelAbstractions
import KernelAbstractions as KA

const HAS_CUDA = try
    @eval using CUDA
    true
catch
    false
end

const HAS_AMD = try
    @eval using AMDGPU
    true
catch
    false
end

function is_cuda_available()
    return HAS_CUDA && !isempty(CUDA.devices())
end

function is_rocm_available()
    return HAS_AMD && !isempty(AMDGPU.devices())
end

function select_backend()
    if is_cuda_available()
        dev = first(CUDA.devices())
        println("using CUDA backend: $(CUDA.name(dev))")
        return CUDABackend()
    elseif is_rocm_available()
        dev = first(AMDGPU.devices())
        println("using AMD backend: $(AMDGPU.HIP.name(dev))")
        return ROCBackend()
    else
        println("using CPU backend")
        return CPU()
    end
end

if is_cuda_available()
    macro call_kernel(name, threads, blocks, args...)
        func_name = name isa Symbol ? Symbol(name, :!) : name
        ex = :( @cuda threads=$threads blocks=$blocks $func_name($(args...)) )
        return esc(ex)
    end
elseif is_rocm_available()
    macro call_kernel(name, threads, blocks, args...)
        func_name = name isa Symbol ? Symbol(name, :!) : name
        ex = :( AMDGPU.@roc groupsize=$threads gridsize=$blocks $func_name($(args...)) )
        return esc(ex)
    end
else
    macro call_kernel(args...)
        error("No GPU backend available to compile @call_kernel")
    end
end

const dev = select_backend()

f(x) = 2*sin(sum(x))

# 0-form for analytical solution
function cpu_loop_1(dΩ_faces)
    s = 0.0
    for dΩ_face in dΩ_faces
        for dΩ_point in GT.each_point_new(dΩ_face)
            x = GT.coordinate(dΩ_point)
            dx = GT.weight(dΩ_point)
            s += f(x)*dx
        end
    end
    return s
end

# 0-form for discrete field
function cpu_loop_2(uh_faces)
    s = 0.0
    for uh_face in uh_faces
        for uh_point in GT.each_point_new(uh_face)
            ux = GT.field(GT.value,uh_point)
            dx = GT.weight(uh_point)
            s += ux*dx
        end
    end
    return s
end

# 0-form for discrete field but with a syntax that potentially
# reuses the jacobians.
function cpu_loop_3(uh_faces,dΩ_faces)
    s = 0.0
    for dΩ_face in dΩ_faces
        uh_face = uh_faces[dΩ_face]
        for dΩ_point in GT.each_point_new(dΩ_face)
            uh_point = uh_face[dΩ_point]
            ux = GT.field(GT.value,uh_point)
            dx = GT.weight(dΩ_point)
            s += ux*dx
        end
    end
    return s
end

# 1-form equivalent to matrix free
function cpu_loop_4!(b,bf,uh_faces)
    for uh_face in uh_faces
        dofs = GT.dofs(uh_face)
        n = GT.num_dofs(uh_face)
        fill!(bf,0)
        for uh_point in GT.each_point_new(uh_face)
            ux = GT.field(GT.gradient,uh_point)
            sx = GT.shape_functions(GT.gradient,uh_point)
            dx = GT.weight(uh_point)
            ux_dx = ux*dx
            for i in 1:n
                bf[i] += ux_dx⋅sx[i]
            end
        end
        for i in 1:n
            b[dofs[i]] += bf[i]
        end
    end
end

# 1-form equivalent to matrix free
# where the discrete field
# and trial functions are potentially different
function cpu_loop_5!(b,bf,uh_faces,V_faces,dΩ_faces)
    for dΩ_face in dΩ_faces
        V_face = V_faces[dΩ_face]
        uh_face = uh_faces[dΩ_face]
        dofs = GT.dofs(V_face)
        n = GT.num_dofs(V_face)
        fill!(bf,0)
        for dΩ_point in GT.each_point_new(dΩ_face)
            V_point = V_face[dΩ_point]
            uh_point = uh_face[dΩ_point]
            ux = GT.field(GT.gradient,uh_point)
            sx = GT.shape_functions(GT.gradient,V_point)
            dx = GT.weight(dΩ_point)
            ux_dx = ux*dx
            for i in 1:n
                bf[i] += ux_dx⋅sx[i]
            end
        end
        for i in 1:n
            b[dofs[i]] += bf[i]
        end
    end
end

# 2-form assembly in coo format
function cpu_loop_6_count(V_faces)
    num_nz = 0
    for V_face in V_faces
        n = GT.num_dofs(V_face)
        num_nz += n*n
    end
    num_nz
end

function cpu_loop_6_symbolic!(AI,AJ,V_faces)
    num_nz = 0
    for V_face in V_faces
        n = GT.num_dofs(V_face)
        dofs = GT.dofs(V_face)
        for j in 1:n
            gj = dofs[j]
            for i in 1:n
                gi = dofs[i]
                num_nz += 1
                AI[num_nz] = gi
                AJ[num_nz] = gj
            end
        end
    end
end

function cpu_loop_6_numeric!(AV,Af,V_faces)
    num_nz = 0
    for V_face in V_faces
        n = GT.num_dofs(V_face)
        fill!(Af,0)
        for V_point in GT.each_point_new(V_face)
            dx = GT.weight(V_point)
            sx = GT.shape_functions(GT.gradient,V_point)
            for j in 1:n
                sx_dx_j = sx[j]*dx
                for i in 1:n
                    Af[i,j] += sx[i]⋅sx_dx_j
                end
            end
        end
        for j in 1:n
            for i in 1:n
                num_nz += 1
                AV[num_nz] = Af[i,j]
            end
        end
    end
end

function main_cpu(params)
    (;face_nodes_layout,face_dofs_layout) = params

    # Start at CPU
    domain = (0,1,0,1)
    cells = (4,4)
    mesh = GT.cartesian_mesh(domain,cells)
    Ω = GT.interior(mesh)
    degree = 4
    dΩ = GT.quadrature(Ω,degree)

    k = 1
    V = GT.lagrange_space(Ω,k)
    u = GT.analytical_field(f,Ω)
    uh = GT.interpolate(u,V)

    tabulate = (GT.value,GT.gradient)
    dΩ_faces_cpu = GT.each_face_new(dΩ)
    V_faces_cpu = GT.each_face_new(V,dΩ;tabulate)
    uh_faces_cpu = GT.each_face_new(uh,dΩ;tabulate)

    # Change data layout
    dΩ_faces_cpu = GT.change_data_layout(dΩ_faces_cpu;face_nodes_layout)
    V_faces_cpu = GT.change_data_layout(V_faces_cpu;face_nodes_layout,face_dofs_layout)
    uh_faces_cpu = GT.change_data_layout(uh_faces_cpu;face_nodes_layout,face_dofs_layout)

    # This is not needed in practice, just to make sure that
    # we do not break anything when adapting.
    dΩ_faces_cpu = Adapt.adapt_structure(Array,dΩ_faces_cpu)
    V_faces_cpu = Adapt.adapt_structure(Array,V_faces_cpu)
    uh_faces_cpu = Adapt.adapt_structure(Array,uh_faces_cpu)

    #dΩ_face_gpu = CUDA.cu(dΩ_faces_cpu)
    #dΩ_face_gpu = GT.loop_options(dΩ_face_gpu;
    #    face_dofs_layout=:face_major, # :face_minor
    #    granularity=:face_per_thread, # :face_per_block
    #    shape_functions_location =:global_memory, # :shared_memory, :kernel_memory
    #   )

    @show r_cpu = cpu_loop_1(dΩ_faces_cpu)
    @show r_cpu = cpu_loop_2(uh_faces_cpu)
    @show r_cpu = cpu_loop_3(uh_faces_cpu,dΩ_faces_cpu)

    b = zeros(GT.num_free_dofs(V))
    nmax = GT.max_num_reference_dofs(V)
    bf = zeros(nmax)
    r_cpu = cpu_loop_4!(b,bf,uh_faces_cpu)
    @show norm(b)

    fill!(b,0)
    r_cpu = cpu_loop_5!(b,bf,uh_faces_cpu,V_faces_cpu,dΩ_faces_cpu)
    @show norm(b)

    num_nz = cpu_loop_6_count(V_faces_cpu)
    AI = zeros(Int32,num_nz)
    AJ = zeros(Int32,num_nz)
    AV = zeros(num_nz)
    Af = zeros(nmax,nmax)
    cpu_loop_6_symbolic!(AI,AJ,V_faces_cpu)
    cpu_loop_6_numeric!(AV,Af,V_faces_cpu)
    n_global = GT.num_dofs(V)
    A,Acache = PA.sparse_matrix(AI,AJ,AV,n_global,n_global;reuse=Val(true))
    x = GT.free_values(uh)
    b = A*x
    @show norm(b)

    # Loop using a previously built matrix
    fill(AV,0)
    cpu_loop_6_numeric!(AV,Af,V_faces_cpu)
    PA.sparse_matrix!(A,AV,Acache)
    b = A*x
    @show norm(b)

end

function cuda_loop_1!(contributions,dΩ_faces_gpu)
    face_id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if face_id > length(dΩ_faces_gpu)
        return nothing
    end
    dΩ_face = dΩ_faces_gpu[face_id]
    s = 0.0
    for dΩ_point in GT.each_point_new(dΩ_face)
        x = GT.coordinate(dΩ_point)
        dx = GT.weight(dΩ_point)
        s += f(x)*dx
    end
    contributions[face_id] = s
    return nothing
end

function hip_loop_1!(contributions,dΩ_faces_gpu)
    face_id = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    if face_id > length(dΩ_faces_gpu)
        return nothing
    end
    dΩ_face = dΩ_faces_gpu[face_id]
    s = 0.0
    for dΩ_point in GT.each_point_new(dΩ_face)
        x = GT.coordinate(dΩ_point)
        dx = GT.weight(dΩ_point)
        s += f(x)*dx
    end
    contributions[face_id] = s
    return nothing
end

@kernel function gpu_loop_1!(contributions,dΩ_faces_gpu)
    face_id = @index(Global)
    if face_id <= length(dΩ_faces_gpu)
        dΩ_face = dΩ_faces_gpu[face_id]
        s = 0.0
        for dΩ_point in GT.each_point_new(dΩ_face)
            x = GT.coordinate(dΩ_point)
            dx = GT.weight(dΩ_point)
            s += f(x)*dx
        end
        contributions[face_id] = s
    end
end

function cuda_loop_2!(contributions,uh_faces_gpu)
    face_id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if face_id > length(uh_faces_gpu)
        return nothing
    end
    uh_face = uh_faces_gpu[face_id]
    s = 0.0
    for uh_point in GT.each_point_new(uh_face)
        ux = GT.field(GT.value,uh_point)
        dx = GT.weight(uh_point)
        s += ux*dx
    end
    contributions[face_id] = s
    return nothing
end

function cuda_loop_5!()
    #TODO
end

function main_gpu(params)
    (;face_nodes_layout,face_dofs_layout) = params

    # Start at CPU
    domain = (0,1,0,1)
    cells = (4,4)
    mesh = GT.cartesian_mesh(domain,cells)
    Ω = GT.interior(mesh)
    degree = 4
    dΩ = GT.quadrature(Ω,degree)

    k = 1
    V = GT.lagrange_space(Ω,k)
    u = GT.analytical_field(f,Ω)
    uh = GT.interpolate(u,V)

    tabulate = (GT.value,GT.gradient)
    dΩ_faces_cpu = GT.each_face_new(dΩ)
    V_faces_cpu = GT.each_face_new(V,dΩ;tabulate)
    uh_faces_cpu = GT.each_face_new(uh,dΩ;tabulate)

    # Change data layout
    # This can also be done after moving memory to GPU (but not implemented yet)
    dΩ_faces_cpu = GT.change_data_layout(dΩ_faces_cpu;face_nodes_layout)
    V_faces_cpu = GT.change_data_layout(V_faces_cpu;face_nodes_layout,face_dofs_layout)
    uh_faces_cpu = GT.change_data_layout(uh_faces_cpu;face_nodes_layout,face_dofs_layout)

    # This is not needed in practice, just to make sure that
    # we do not break anything when adapting.
    if is_cuda_available()
        dΩ_faces_gpu = CUDA.cu(dΩ_faces_cpu)
        V_faces_gpu = CUDA.cu(V_faces_cpu)
        uh_faces_gpu = CUDA.cu(uh_faces_cpu)
    end

    #TODO
    #granularity = Val(:face_per_thread) # Val(:face_per_block)
    #dΩ_faces_cpu = GT.change_loop_granularity(dΩ_faces_cpu,granularity)
    #V_faces_cpu = GT.change_loop_granularity(V_faces_cpu,granularity)
    #uh_faces_cpu = GT.change_loop_granularity(uh_faces_cpu,granularity)
    #
    #Maybe workspace location should not be independent and depend on granularity
    #workspace_location = Val(:global_memory) # Val(:shared_memory) Val(:thread_memory)
    #dΩ_faces_gpu = GT.change_workspace_location(dΩ_faces_gpu,workspace_location)
    #V_faces_gpu = GT.change_workspace_location(V_faces_gpu,workspace_location)
    #uh_faces_gpu = GT.change_workspace_location(uh_faces_gpu,workspace_location)

    nfaces = length(dΩ_faces_gpu)
    contributions = KA.zeros(dev, Float64, nfaces)
    r_gpu = 0

    # Launch kernel 1
    threads_in_block = 256
    t1_gpu = @benchmark begin
        gpu_loop_1!($dev, $threads_in_block)($contributions, $dΩ_faces_gpu, ndrange=$nfaces)
        global r_gpu = sum($contributions)
        KA.synchronize($dev)
    end
    @show r_gpu

    if is_cuda_available()
        # Launch kernel 1
        threads_in_block = 256
        blocks_in_grid = cld(nfaces, threads_in_block)
        t1_cuda = @benchmark begin
            @call_kernel cuda_loop_1 $threads_in_block $blocks_in_grid $contributions $dΩ_faces_gpu
            global r_gpu = sum($contributions)
            CUDA.synchronize()
        end
        @show r_gpu  
        # Launch kernel 2
        # threads_in_block = 256
        # blocks_in_grid = cld(nfaces, threads_in_block)
        # t2_cuda = @benchmark begin
        #     @call_kernel cuda_loop_2 $threads_in_block $blocks_in_grid $contributions $uh_faces_gpu
        #     global r_gpu = sum($contributions)
        #     CUDA.synchronize()
        # end
        # @show r_gpu
    elseif is_rocm_available()
        # Launch kernel 1
        threads_in_block = 256
        blocks_in_grid = cld(nfaces, threads_in_block)
        t1_hip = @benchmark begin
            @call_kernel hip_loop_1 $threads_in_block $blocks_in_grid $contributions $dΩ_faces_gpu
            global r_gpu = sum($contributions)
            AMDGPU.synchronize()
        end
        @show r_gpu
    end

    println("Loop 1: KernelAbstractions throughput is ", nfaces / time(t1_gpu) * 1e9, " faces per second.")
    if is_cuda_available()
        println("Loop 1: CUDA throughput is ", nfaces / time(t1_cuda) * 1e9, " faces per second.")
        println("Loop 1: CUDA speedup is ", (nfaces / time(t1_cuda) * 1e9) / (nfaces / time(t1_gpu) * 1e9))
        # println("Loop 2: CUDA throughput is ", nfaces / time(t2_cuda) * 1e9, " faces per second.")
    elseif is_rocm_available()
        println("Loop 1: HIP throughput is ", nfaces / time(t1_hip) * 1e9, " faces per second.")
        println("Loop 1: HIP speedup is ", (nfaces / time(t1_hip) * 1e9) / (nfaces / time(t1_gpu) * 1e9))
    end
end

layouts = (GT.face_minor_array,GT.face_major_array)
for face_dofs_layout in layouts
    for face_nodes_layout in layouts
        params = (;face_nodes_layout,face_dofs_layout)
        main_cpu(params)
        main_gpu(params)
        println()
    end
end

end # module
