// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/normalization.h"
#include "../extensions.h"

namespace primus_turbo::jax {

// TODO: remove
// inline void PrintBufferInfo(const char *name, const ffi::AnyBuffer &buf) {
//     printf("%s:\n", name);
//     printf("Shape: ");
//     const ffi::AnyBuffer::Dimensions dims = buf.dimensions();
//     for (auto d : dims) {
//         printf("%d ", d);
//     }
//     printf("\n");
// }

ffi::Error RMSNormFwdFFI(cudaStream_t stream, ffi::AnyBuffer input, ffi::AnyBuffer gamma,
                         ffi::Result<ffi::AnyBuffer> output, double eps) {
    const int64_t inner_len = static_cast<int64_t>(gamma.element_count());
    const int64_t outer_len = static_cast<int64_t>(input.element_count() / inner_len);
    const float   eps_f     = static_cast<float>(eps);

    switch (input.element_type()) {
        case ffi::F16:
            rmsnorm_fwd_impl<float16>(input.typed_data<float16>(), gamma.typed_data<float16>(),
                                     output->typed_data<float16>(), inner_len, outer_len, eps_f,
                                     stream);
            break;
        case ffi::BF16:
            rmsnorm_fwd_impl<bfloat16>(input.typed_data<bfloat16>(), gamma.typed_data<bfloat16>(),
                                      output->typed_data<bfloat16>(), inner_len, outer_len, eps_f,
                                      stream);
            break;
        default:
            rmsnorm_fwd_impl<float>(input.typed_data<float>(), gamma.typed_data<float>(),
                                   output->typed_data<float>(), inner_len, outer_len, eps_f,
                                   stream);
    }
    return ffi::Error::Success();
}

ffi::Error RMSNormBwdFFI(cudaStream_t stream, ffi::AnyBuffer doutput, ffi::AnyBuffer input,
                         ffi::AnyBuffer gamma, ffi::Result<ffi::AnyBuffer> dinput,
                         ffi::Result<ffi::AnyBuffer> dgamma, double eps) {
    const int64_t inner_len = static_cast<int64_t>(gamma.element_count());
    const int64_t outer_len = static_cast<int64_t>(input.element_count() / inner_len);
    const float   eps_f     = static_cast<float>(eps);

    switch (input.element_type()) {
        case ffi::F16:
            rmsnorm_bwd_impl<float16>(input.typed_data<float16>(), gamma.typed_data<float16>(),
                                     doutput.typed_data<float16>(), dinput->typed_data<float16>(),
                                     dgamma->typed_data<float16>(), inner_len, outer_len, eps_f,
                                     stream);
            break;
        case ffi::BF16:
            rmsnorm_bwd_impl<bfloat16>(input.typed_data<bfloat16>(), gamma.typed_data<bfloat16>(),
                                      doutput.typed_data<bfloat16>(), dinput->typed_data<bfloat16>(),
                                      dgamma->typed_data<bfloat16>(), inner_len, outer_len, eps_f,
                                      stream);
            break;
        default:
            rmsnorm_bwd_impl<float>(input.typed_data<float>(), gamma.typed_data<float>(),
                                   doutput.typed_data<float>(), dinput->typed_data<float>(),
                                   dgamma->typed_data<float>(), inner_len, outer_len, eps_f,
                                   stream);
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(RMSNormFwdHandler, RMSNormFwdFFI,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>() // stream
                                  .Arg<ffi::AnyBuffer>()                    // input
                                  .Arg<ffi::AnyBuffer>()                    // gamma
                                  .Ret<ffi::AnyBuffer>()                    // output
                                  .Attr<double>("eps")                      // eps
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(RMSNormBwdHandler, RMSNormBwdFFI,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>() // stream
                                  .Arg<ffi::AnyBuffer>()                    // doutput
                                  .Arg<ffi::AnyBuffer>()                    // input
                                  .Arg<ffi::AnyBuffer>()                    // gamma
                                  .Ret<ffi::AnyBuffer>()                    // dinput
                                  .Ret<ffi::AnyBuffer>()                    // dgamma
                                  .Attr<double>("eps"));

ffi::Error AdaRMSNormFwdFFI(cudaStream_t stream, ffi::AnyBuffer input, ffi::AnyBuffer gamma,
                             ffi::AnyBuffer ada_scale, ffi::AnyBuffer ada_shift,
                             ffi::Result<ffi::AnyBuffer> output, double eps) {
    const int64_t inner_len = static_cast<int64_t>(gamma.element_count());
    const int64_t outer_len = static_cast<int64_t>(input.element_count() / inner_len);
    const float   eps_f     = static_cast<float>(eps);

    switch (input.element_type()) {
        case ffi::F16:
            adarmsnorm_fwd_impl<float16>(
                input.typed_data<float16>(), gamma.typed_data<float16>(),
                ada_scale.typed_data<float16>(), ada_shift.typed_data<float16>(),
                output->typed_data<float16>(), inner_len, outer_len, eps_f, stream);
            break;
        case ffi::BF16:
            adarmsnorm_fwd_impl<bfloat16>(
                input.typed_data<bfloat16>(), gamma.typed_data<bfloat16>(),
                ada_scale.typed_data<bfloat16>(), ada_shift.typed_data<bfloat16>(),
                output->typed_data<bfloat16>(), inner_len, outer_len, eps_f, stream);
            break;
        default:
            adarmsnorm_fwd_impl<float>(
                input.typed_data<float>(), gamma.typed_data<float>(),
                ada_scale.typed_data<float>(), ada_shift.typed_data<float>(),
                output->typed_data<float>(), inner_len, outer_len, eps_f, stream);
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(AdaRMSNormFwdHandler, AdaRMSNormFwdFFI,
                               ffi::Ffi::Bind()
                                   .Ctx<ffi::PlatformStream<cudaStream_t>>() // stream
                                   .Arg<ffi::AnyBuffer>()                    // input
                                   .Arg<ffi::AnyBuffer>()                    // gamma
                                   .Arg<ffi::AnyBuffer>()                    // ada_scale
                                   .Arg<ffi::AnyBuffer>()                    // ada_shift
                                   .Ret<ffi::AnyBuffer>()                    // output
                                   .Attr<double>("eps")                      // eps
);

} // namespace primus_turbo::jax
