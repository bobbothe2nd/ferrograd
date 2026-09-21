use crate::{
    dispatch::{
        CompilationOptions, DebugCompilationOptions, GpuBackend, GpuBufferBackend, GpuKernelBackend, TargetCompilationOptions, TargetFlags, backend::{
            MetaId, NodeId, Param,
            kernel::{Dependencies, RawKernel, Redirect},
        },
    }, errors::{Error, ErrorKind}, tensor::{ToBuffer, build_dims, calc_grid},
};
use briny::raw::cast::cast_slice;
use std::vec::Vec;

pub use wgpu::{
    BackendOptions, Backends, BindGroup, BindGroupDescriptor, BindGroupEntry,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType,
    BufferDescriptor, BufferUsages, CommandBuffer, CommandEncoder, CommandEncoderDescriptor,
    ComputePass, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Device,
    DeviceDescriptor, Dx12BackendOptions, Dx12Compiler, Dx12SwapchainKind,
    Dx12UseFrameLatencyWaitableObject, ExperimentalFeatures, Features, ForceShaderModelToken,
    GlBackendOptions, GlDebugFns, GlFenceBehavior, Gles3MinorVersion, Instance, InstanceDescriptor,
    InstanceFlags, Limits, MemoryBudgetThresholds, MemoryHints, NoopBackendOptions,
    PipelineCompilationOptions, PipelineLayout, PipelineLayoutDescriptor, PollType,
    PowerPreference, Queue, RequestAdapterError, RequestAdapterOptions, ShaderModuleDescriptor,
    ShaderSource, ShaderStages, SubmissionIndex, Trace, PollStatus,
    util::{BufferInitDescriptor, DeviceExt},
};

mod generate;

/// WGPU context for device and queue.
#[derive(Debug)]
pub struct GpuContext {
    device: Device,
    queue: Queue,
}

impl GpuContext {
    /// Constructs a new context asynchronously.
    pub async fn new() -> Result<Self, Error> {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            flags: InstanceFlags::empty(),
            memory_budget_thresholds: MemoryBudgetThresholds::default(),
            backend_options: BackendOptions {
                gl: GlBackendOptions {
                    gles_minor_version: Gles3MinorVersion::Automatic,
                    fence_behavior: GlFenceBehavior::AutoFinish,
                    debug_fns: GlDebugFns::Disabled,
                },
                dx12: Dx12BackendOptions {
                    shader_compiler: Dx12Compiler::Auto,
                    presentation_system: Dx12SwapchainKind::default(),
                    latency_waitable_object: Dx12UseFrameLatencyWaitableObject::None,
                    force_shader_model: ForceShaderModelToken::default(),
                    agility_sdk: None,
                },
                noop: NoopBackendOptions::default(),
            },
            display: None,
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|e| match e {
                RequestAdapterError::EnvNotSet => Error {
                    msg: "adapter environment variables not set",
                    kind: ErrorKind::EnvNotSet,
                    ctx: (),
                },
                _ => Error {
                    msg: "no adapter found",
                    kind: ErrorKind::InvalidDevice,
                    ctx: (),
                },
            })?;

        let adapter_limits = adapter.limits();

        let (device, queue) = match adapter
            .request_device(&DeviceDescriptor {
                label: Some("device"),
                required_limits: adapter_limits,
                required_features: Features::SHADER_F16,
                experimental_features: ExperimentalFeatures::disabled(),
                memory_hints: MemoryHints::Performance,
                trace: Trace::Off,
            })
            .await
        {
            Ok(dev) => dev,
            Err(_) => adapter
                .request_device(&DeviceDescriptor {
                    label: Some("device"),
                    required_limits: Limits::defaults(),
                    required_features: Features::SHADER_F16,
                    experimental_features: ExperimentalFeatures::disabled(),
                    memory_hints: MemoryHints::Performance,
                    trace: Trace::Off,
                })
                .await
                .map_err(|_| Error {
                    msg: "failed to request device",
                    kind: ErrorKind::InvalidDevice,
                    ctx: (),
                })?,
        };

        Ok(Self { device, queue })
    }
}

#[derive(Debug, Clone)]
pub struct GpuKernel {
    kernel: ComputePipeline,
    iter_space: Vec<MetaId>,
    block: [u32; 3],
}

impl GpuKernelBackend for GpuKernel {
    fn iteration_space(&self) -> &[MetaId] {
        &self.iter_space
    }

    fn block(&self) -> &[u32; 3] {
        &self.block
    }
}

pub struct Schedule {
    kernels: Vec<(ComputePipeline, [u32; 3], BindGroup)>,
}

impl GpuBackend for GpuContext {
    type Buffer = Buffer;
    type MetaBuf = Buffer;
    type Kernel = GpuKernel;
    type Schedule = Schedule;

    fn target_spec(&self) -> TargetCompilationOptions {
        TargetCompilationOptions {
            flags: TargetFlags::empty(),
        }
    }

    #[inline]
    fn alloc(&self, len: u32) -> Result<Self::Buffer, Error> {
        Ok(self.device.create_buffer(&BufferDescriptor {
            label: Some("gpu_tensor"),
            size: len as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }))
    }

    #[inline]
    fn alloc_init(&self, contents: &[u8]) -> Result<Self::Buffer, Error> {
        Ok(self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("gpu_tensor"),
            contents,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        }))
    }

    #[inline]
    fn alloc_meta(&self, data: &[u32]) -> Result<Self::Buffer, Error> {
        let offset = (4 - (data.len() % 4)) % 4;
        let new_len = data.len() + offset;
        let mut aligned = std::vec![u32::MAX; new_len];
        aligned[..data.len()].copy_from_slice(data);

        Ok(self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("gpu_meta"),
            contents: cast_slice(&aligned),
            usage: BufferUsages::UNIFORM,
        }))
    }

    #[inline]
    fn upload(
        &self,
        buffer: &Self::Buffer,
        data: &[u8],
        src_off: u32,
        dst_off: u32,
    ) -> Result<(), Error> {
        if buffer.size_bytes().saturating_sub(dst_off) as usize > data.len() {
            return Err(Error {
                msg: "CPU buffer of smaller size than GPU buffer during upload",
                kind: ErrorKind::FailedBufferCopy,
                ctx: (),
            });
        }

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        let src = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("gpu_tensor"),
            contents: data,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        encoder.copy_buffer_to_buffer(&src, src_off as u64, &buffer, dst_off as u64, Some(data.len() as u64));

        self.queue.submit(Some(encoder.finish()));

        Ok(())
    }

    fn copy(&self, src: &Self::Buffer, dst: &Self::Buffer) -> Result<(), Error> {
        let src_size = src.size();

        if src_size != dst.size() {
            return Err(Error {
                msg: "buffers of unequal sizes during pipe",
                kind: ErrorKind::FailedBufferCopy,
                ctx: (),
            });
        }

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());

        encoder.copy_buffer_to_buffer(&src, 0, &dst, 0, src_size);

        self.queue.submit(Some(encoder.finish()));

        Ok(())
    }

    #[inline]
    fn download(&self, buffer: &Self::Buffer, data: &mut [u8]) -> Result<(), Error> {
        if buffer.size_bytes() as usize > data.len() {
            return Err(Error {
                msg: "insufficient CPU memory allocated for GPU download",
                kind: ErrorKind::FailedBufferCopy,
                ctx: (),
            });
        }

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        let dst = self.device.create_buffer(&BufferDescriptor {
            label: Some("download"),
            size: buffer.size(),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&buffer, 0, &dst, 0, buffer.size());
        let submission_index = self.queue.submit(Some(encoder.finish()));
        let buffer_slice = dst.slice(..);

        let (send, recv) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
            if res.is_ok() {
                let _ = send.send(());
            }
        });

        let _ = self.device.poll(PollType::Wait {
            submission_index: Some(submission_index),
            timeout: None,
        });

        let _ = recv.recv();

        let len = data.len().min(buffer.size_bytes() as usize);

        data[..len].copy_from_slice(
            &buffer_slice.get_mapped_range().map_err(|_| Error {
                msg: "failed to map GPU memory to CPU",
                kind: ErrorKind::FailedBufferCopy,
                ctx: (),
            })?[..len],
        );
        dst.unmap();

        Ok(())
    }

    #[inline]
    fn compile(
        &self,
        src: &RawKernel,
        params: &[Param],
        options: &CompilationOptions,
    ) -> Result<Self::Kernel, Error> {
        let entries = generate::generate_layout_desc(params);
        let desc = BindGroupLayoutDescriptor {
            label: Some("bind_group_layout"),
            entries: &entries,
        };

        println!("{:?}\n", entries);

        let bind_group_layout = self.device.create_bind_group_layout(&desc);
        let bind_group_layouts = &[Some(&bind_group_layout)];

        let desc = PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts,
            immediate_size: 0,
        };
        let pipeline_layout = self.device.create_pipeline_layout(&desc);

        let source = generate::generate_wgsl(
            src,
            params,
            options
                .debug
                .contains(DebugCompilationOptions::PRETTY_PRINT_IR),
        )?;

        let shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("shader"),
            source: ShaderSource::Wgsl(source.into()),
        });

        Ok(GpuKernel {
            kernel: self
                .device
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: Some("pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: PipelineCompilationOptions::default(),
                    cache: None,
                }),
            iter_space: src.iter_space.clone(),
            block: src.block,
        })
    }

    fn schedule(
        &self,
        kernels: Vec<Dependencies<Redirect<(Self::Kernel, NodeId, &[bool])>>>,
        bindings: &[&Self::Buffer],
        meta: &[u32],
        meta_buf: &Self::MetaBuf,
    ) -> Result<Self::Schedule, Error> {
        let mut resolved = Vec::new();
        let mut tmp_res = Vec::new();

        let mut scheduled_kernels = Vec::new();

        while resolved.len() < kernels.len() {
            for kernel in &kernels {
                let dep = &kernel.dep;

                let kernel = match &kernel.val {
                    Redirect::Unmasked(kernel_data) => kernel_data,
                    Redirect::Redirected(idx) => match &kernels[*idx].val {
                        Redirect::Unmasked(kernel) => kernel,
                        Redirect::Redirected(_) => {
                            return Err(Error {
                                msg: "double redirection or loop encountered in kernel resolution",
                                kind: ErrorKind::UnresolvedRedirection,
                                ctx: (),
                            });
                        }
                    },
                };

                let (kernel, idx, params) = kernel;

                if resolved.contains(idx) {
                    continue;
                }

                if dep.iter().all(|x| resolved.contains(x)) {
                    tmp_res.push(*idx);

                    let iter_space = build_dims(kernel.iteration_space(), meta);
                    let grid = calc_grid(&iter_space, *kernel.block());

                    let mut kernel_bindings = Vec::with_capacity(1 + bindings.len());

                    kernel_bindings.push(BindGroupEntry {
                        binding: 0,
                        resource: meta_buf.as_entire_binding(),
                    });

                    bindings
                        .iter()
                        .enumerate()
                        .filter_map(|(i, buf)| {
                            if params[i] {
                                let entry = BindGroupEntry {
                                    binding: 1 + i as u32,
                                    resource: buf.as_entire_binding(),
                                };

                                Some(entry)
                            } else {
                                None
                            }
                        })
                        .for_each(|entry| kernel_bindings.push(entry));

                    let kernel = &kernel.kernel;

                    println!("{:?}\n", kernel_bindings);

                    let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                        layout: &kernel.get_bind_group_layout(0),
                        entries: &kernel_bindings,
                        label: Some("bind_group"),
                    });

                    scheduled_kernels.push((kernel.clone(), grid, bind_group));
                }
            }

            resolved.append(&mut tmp_res);
        }

        Ok(Schedule {
            kernels: scheduled_kernels,
        })
    }

    fn dispatch_schedule(&self, schedule: &Self::Schedule) -> Result<(), Error> {
        let mut encoder = self.device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("pass"),
                timestamp_writes: None,
            });

            for (kernel, wg, bind_group) in &schedule.kernels {
                pass.set_pipeline(kernel);
                pass.set_bind_group(0, bind_group, &[]);

                pass.dispatch_workgroups(wg[0], wg[1], wg[2]);
            }
        }

        self.queue.submit([encoder.finish()]);

        Ok(())
    }

    #[inline]
    fn dispatch_kernel(
        &self,
        kernel: &Self::Kernel,
        wg: [u32; 3],
        bindings: &[&Self::Buffer],
        meta: &Self::MetaBuf,
    ) -> Result<(), Error> {
        let kernel = &kernel.kernel;

        let mut entries = Vec::with_capacity(1 + bindings.len());

        entries.push(BindGroupEntry {
            binding: 0,
            resource: meta.as_entire_binding(),
        });

        bindings
            .iter()
            .enumerate()
            .map(|(i, x)| BindGroupEntry {
                binding: 1 + i as u32,
                resource: x.as_entire_binding(),
            })
            .for_each(|entry| entries.push(entry));

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            layout: &kernel.get_bind_group_layout(0),
            entries: &entries,
            label: Some("bind_group"),
        });

        let mut encoder = self.device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("encoder"),
            });
    
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(kernel);
            pass.set_bind_group(0, &bind_group, &[]);

            pass.dispatch_workgroups(wg[0], wg[1], wg[2]);
        }

        self.queue.submit([encoder.finish()]);

        Ok(())
    }

    #[inline]
    fn sync(&self) -> Result<(), Error> {
        self.device.poll(PollType::Wait {
            submission_index: None,
            timeout: None,
        }).map_err(|_| Error {
            kind: ErrorKind::PollFailed,
            msg: "failed to poll GPU for completion",
            ctx: (),
        })?;

        Ok(())
    }

    fn is_ready(&self) -> Result<bool, Error> {
        let res = self.device.poll(PollType::Poll);

        match res {
            Err(_) => Err(Error {
                msg: "failed to poll device for completion",
                kind: ErrorKind::PollFailed,
                ctx: (),
            }),
            Ok(PollStatus::Poll) => Ok(false),
            Ok(wgpu::PollStatus::WaitSucceeded) => Ok(true),
            Ok(wgpu::PollStatus::QueueEmpty) => Err(Error {
                msg: "queue empty on poll",
                kind: ErrorKind::PollFailed,
                ctx: (),
            }),
        }
    }
}

impl GpuBufferBackend for Buffer {
    #[inline]
    fn size_bytes(&self) -> u32 {
        self.size() as u32
    }

    #[inline]
    fn size(&self) -> u32 {
        (self.size() / (size_of::<f32>() as u64)) as u32
    }
}

impl ToBuffer<GpuContext> for Buffer {
    #[inline]
    fn as_buffer(&self) -> &Self {
        self
    }

    #[inline]
    fn to_buffer(self) -> Self {
        self
    }
}
