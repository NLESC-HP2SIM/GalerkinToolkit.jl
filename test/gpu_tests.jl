module GPUTests

using Test
using BenchmarkTools
using Adapt
using KernelAbstractions
import KernelAbstractions as KA
import GalerkinToolkit as GT

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
    macro call_kernel(name, threads_in_block, blocks_in_grid, args...)
        esc(:(@cuda threads=$threads_in_block blocks=$blocks_in_grid $name\!($(args...))))
    end
elseif is_rocm_available()
    macro call_kernel(name, threads_in_block, blocks_in_grid, args...)
        esc(:(@roc groupsize=$threads_in_block gridsize=$blocks_in_grid $name\!($(args...))))
    end
end

const dev = select_backend()

# Goal integrate function f(x)
f(x) = 2*sin(sum(x))

function cuda_kernel!(contributions,dΩ_faces_gpu)
    face_id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if face_id > length(dΩ_faces_gpu)
        return nothing
    end
    dΩ_face = dΩ_faces_gpu[face_id]
    s = 0.0
    for dΩ_point in GT.each_point(dΩ_face)
        x = GT.coordinate(dΩ_point)
        dx = GT.weight(dΩ_point)
        s += f(x)*dx
    end
    contributions[face_id] = s
    return nothing
end

function hip_kernel!(contributions,dΩ_faces_gpu)
    face_id = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    if face_id > length(dΩ_faces_gpu)
        return nothing
    end
    dΩ_face = dΩ_faces_gpu[face_id]
    s = 0.0
    for dΩ_point in GT.each_point(dΩ_face)
        x = GT.coordinate(dΩ_point)
        dx = GT.weight(dΩ_point)
        s += f(x)*dx
    end
    contributions[face_id] = s
    return nothing
end

@kernel function gpu_kernel!(contributions,dΩ_faces_gpu)
    face_id = @index(Global)
    if face_id <= length(dΩ_faces_gpu)
        dΩ_face = dΩ_faces_gpu[face_id]
        s = 0.0
        for dΩ_point in GT.each_point(dΩ_face)
            x = GT.coordinate(dΩ_point)
            dx = GT.weight(dΩ_point)
            s += f(x)*dx
        end
        contributions[face_id] = s
    end
end

# For reference, this is how you can do this on the CPU
function cpu_loop(dΩ_faces)
    s = 0.0
    for dΩ_face in dΩ_faces
        for dΩ_point in GT.each_point(dΩ_face)
            x = GT.coordinate(dΩ_point)
            dx = GT.weight(dΩ_point)
            s += f(x)*dx
        end
    end
    return s
end

function benchmark_case(;cells::Tuple{Int, Int}, degree::Int)
    # on a domain Ω defined by a computational mesh
    # (a square in this example)
    domain = (0,1,0,1)
    mesh = GT.cartesian_mesh(domain,cells)
    Ω = GT.interior(mesh)
    dΩ = GT.quadrature(Ω,degree)
    dΩ_faces = GT.each_face(dΩ)
    nfaces = length(dΩ_faces)

    # Run on the CPU
    r_cpu = cpu_loop(dΩ_faces)
    b_cpu = @benchmark cpu_loop($dΩ_faces)

    # This is still on CPU, but with a data
    # layout more appropriate for GPUs.
    dΩ_faces_cpu = GT.device_layout(dΩ_faces)

    # Now, move data to GPU
    dΩ_faces_gpu = adapt(dev, dΩ_faces_cpu)
    contributions = KA.zeros(dev,Float64,nfaces)
    # Launch kernel on the GPU
    threads_in_block = 256
    blocks_in_grid = cld(nfaces, threads_in_block)
    if is_cuda_available()
        @call_kernel cuda_kernel threads_in_block blocks_in_grid contributions dΩ_faces_gpu
        r_cuda = sum(contributions)
    elseif is_rocm_available()
        @call_kernel hip_kernel threads_in_block blocks_in_grid contributions dΩ_faces_gpu
        r_hip = sum(contributions)
    end

    if is_cuda_available()
        b_cuda = @benchmark begin
            @call_kernel cuda_kernel $threads_in_block $blocks_in_grid $contributions $dΩ_faces_gpu
            sum($contributions)
            CUDA.synchronize()
        end
    elseif is_rocm_available()
        b_hip = @benchmark begin
            @call_kernel hip_kernel $threads_in_block $blocks_in_grid $contributions $dΩ_faces_gpu
            sum($contributions)
            AMDGPU.synchronize()
       end
    end

    b_gpu = @benchmark begin
        gpu_kernel!($dev, $threads_in_block)($contributions,$dΩ_faces_gpu,ndrange=$nfaces)
        sum($contributions)
        KA.synchronize($dev)
    end

    if is_cuda_available
        @test r_cuda≈r_cpu
        return (
                cells=cells,
                nfaces=nfaces,
                degree=degree,
                throughput_cpu=nfaces / time(b_cpu) * 1e9, # ns - >sec
                throughput_gpu=nfaces / time(b_gpu) * 1e9, # ns -> sec
                throughput_cuda=nfaces / time(b_cuda) * 1e9, # ns -> sec
        )
    elseif is_rocm_available
        @test r_hip≈r_cpu
        return (
                cells=cells,
                nfaces=nfaces,
                degree=degree,
                throughput_cpu=nfaces / time(b_cpu) * 1e9, # ns - >sec
                throughput_gpu=nfaces / time(b_gpu) * 1e9, # ns -> sec
                throughput_hip=nfaces / time(b_hip) * 1e9, # ns -> sec
        )
    end

end


for k in [2, 10, 25, 100, 250, 500, 1000, 2500]
    println(benchmark_case(cells=(k, k), degree=4))
end


end # module
