/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif


#include "caffe/layers/conv_layer.hpp"

#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  // LOG(ERROR) << "input: [" << bottom[0]->num() << ", " << bottom[0]->channels() << ", " << bottom[0]->height() << ", " << bottom[0]->width() << "]";
  // LOG(ERROR) << "output: [" << top[0]->num() << ", " << top[0]->channels() << ", " << top[0]->height() << ", " << top[0]->width() << "]";
  // LOG(ERROR) << "filter: [" << this->kernel_shape_.cpu_data()[0] << ", " << this->kernel_shape_.cpu_data()[1] << "]";
  // LOG(ERROR) << "stride: [" << this->stride_.cpu_data()[0] << ", " << this->stride_.cpu_data()[1] << "]";
  // LOG(ERROR) << "dilation: [" << this->dilation_.cpu_data()[0] << ", " << this->dilation_.cpu_data()[1] << "]";

  // Timer timer, timer2;
  // timer.Start();

  const Dtype* weight = this->blobs_[0]->cpu_data();
  // If we have more threads available than batches to be prcessed then
  // we are wasting resources (lower batches than 36 on XeonE5)
  // So we instruct MKL
  // LOG(ERROR) << "bottom size: " << bottom.size();
  for (int i = 0; i < bottom.size(); ++i) {
    // timer2.Start();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    // LOG(ERROR) << "get data takes: " << timer2.MicroSeconds() / 1000. << " ms";
    // timer2.Start();
// #ifdef _OPENMP
//    #pragma omp parallel for num_threads(this->num_of_threads_)
// #endif
      for (int n = 0; n < this->num_; ++n) {
        this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_,
                               weight,
                               top_data + n * this->top_dim_);
        // LOG(ERROR) << "mkl thread number: " << omp_get_max_threads();
        if (this->bias_term_) {
          const Dtype* bias = this->blobs_[1]->cpu_data();
          this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
        }
      }
      // LOG(ERROR) << "forward GEMM takes: " << timer2.MicroSeconds() / 1000. << " ms";
  }

  // dump conv output
#if 0
  static int cnt = 0;
  if (!this->layer_param_.name().compare("rpn_conv/3x3") && cnt == 0) {
    FILE *fp = fopen("./rpn_conv_cpu.txt", "wb");
    const Dtype* top_data = top[0]->cpu_data();
    int i = 0;
    for (int n = 0; n < top[0]->num(); n++) {
      for (int c = 0; c < 1; c++) {
        for (int h = 0; h < top[0]->height(); h++) {
          for (int w = 0; w < top[0]->width(); w++) {
            fprintf(fp, "%.2f, ", top_data[i]);
            i++;
          }
        }
      }
    }
   fclose(fp);

   // print weights
   FILE *fp = fopen("./rpn_conv_cpu_weights.txt", "wb");
   for (int n = 0; n < this->blobs_[0].count(); n++) {
      fprintf(fp, "%.2f, ", this->blobs_[0]->cpu_data()[n]);
   }
   fclose(fp);
  }
  cnt++;
#endif

  // LOG(ERROR) << "forward total takes: " << timer.MicroSeconds() / 1000. << " ms";
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // Timer timer;
  // timer.Start();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  // LOG(ERROR) << "get weight data takes: " << timer.MicroSeconds() / 1000. << " ms";
  // timer.Start();
  for (int i = 0; i < top.size(); ++i) {
    // Timer timer2;
    // timer2.Start();
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // LOG(ERROR) << "get data takes: " << timer2.MicroSeconds() / 1000. << " ms";
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }

    // OpenMP path is using bigger separate buffer to accumulate
    // weight diffs, which are lateron add to weight_diff
    // so bigger buffer (weight_diff_mt) hase to be cleared out
    // before GEMM ops and results has to be summed up after GEMM ops.

    // timer2.Start();
    if (this->param_propagate_down_[0]) {
// #ifdef _OPENMP
//      this->clear_weight_mt();
//      #pragma omp parallel num_threads(this->num_of_threads_)
// #endif
      {
// #ifdef _OPENMP
//         #pragma omp for
// #endif
        for (int n = 0; n < this->num_; ++n) {
          // gradient w.r.t. weight. Note that we will accumulate diffs.
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
                top_diff + n * this->top_dim_, weight_diff);
        }

// #ifdef _OPENMP
//        this->sum_weight_mt(weight_diff);
// #endif
      }
    }
    // LOG(ERROR) << "weight back propagation takes: " << timer2.MicroSeconds() / 1000. << " ms";
    // timer2.Start();
	// LOG(ERROR) << this->layer_param_.name() << " blob: " << i << " propagate down: " << propagate_down[i];
    if (propagate_down[i]) {
// #ifdef _OPENMP
//       #pragma omp parallel for num_threads(this->num_of_threads_)
// #endif
        for (int n = 0; n < this->num_; ++n) {
          // gradient w.r.t. bottom data, if necessary.
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
    }
   // LOG(ERROR) << "data back propagation takes: " << timer2.MicroSeconds() / 1000. << " ms";
  }

  // LOG(ERROR) << "backward total takes: " << timer.MicroSeconds() / 1000. << " ms";
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
