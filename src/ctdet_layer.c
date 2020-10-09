//
// Created by yoloCao on 19-8-26.
//

//
// Created by yoloCao on 19-8-14.
//

#include "ctdet_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "dark_cuda.h"
#include "utils.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

layer make_ctdet_layer(int batch, int w, int h , int classes, int size ,int stride,int padding)
{
    layer l = { (LAYER_TYPE)0 };
    l.type = CTDET;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = (4+classes);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.cost = (float*)xcalloc(1, sizeof(float));
    l.outputs = h*w*(4+classes);
    l.inputs = l.outputs;
    l.truths = h*w*(4+classes);
    l.delta = (float*)xcalloc(batch*l.outputs, sizeof(float));
    l.output = (float*)xcalloc(batch*l.outputs, sizeof(float));

    l.pad = padding;
    l.size = size;
    l.stride = stride;
    l.indexes = (int*)xcalloc(batch*l.outputs, sizeof(int));

    l.num_detection = (int*)xcalloc(1, sizeof(int));
    l.forward = forward_ctdet_layer;
    l.backward = backward_ctdet_layer;

#ifdef GPU
    l.num_detection_gpu = cuda_make_int_array_ctdet(0,1);
    l.forward_gpu = forward_ctdet_layer_gpu;
    l.backward_gpu = backward_ctdet_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
    l.indexes_gpu = cuda_make_int_array_ctdet(l.indexes, batch*l.outputs);

    free(l.output);
    if (cudaSuccess == cudaHostAlloc(&l.output, batch*l.outputs*sizeof(float), cudaHostRegisterMapped)) l.output_pinned = 1;
    else { 
        cudaGetLastError(); // reset CUDA-error
        l.output = (float*)xcalloc(batch * l.outputs, sizeof(float));
    }

    free(l.delta);
    if (cudaSuccess == cudaHostAlloc(&l.delta, batch*l.outputs*sizeof(float), cudaHostRegisterMapped)) l.delta_pinned = 1;
    else { 
        cudaGetLastError(); // reset CUDA-error
        l.delta = (float*)xcalloc(batch * l.outputs, sizeof(float));
    }
#endif
    
    fprintf(stderr, "ctdet ");
    srand(time(0));

    return l;
    
}

void resize_ctdet_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*(4+l->classes);
    l->inputs = l->outputs;

    if (!l->output_pinned) l->output = (float*)xrealloc(l->output, l->batch*l->outputs * sizeof(float));
    if (!l->delta_pinned) l->delta = (float*)xrealloc(l->delta, l->batch*l->outputs*sizeof(float));
    l->indexes = (float*)xrealloc(l->indexes, l->batch*l->outputs*sizeof(int));

#ifdef GPU

    if (l->output_pinned) {
        CHECK_CUDA(cudaFreeHost(l->output));
        if (cudaSuccess != cudaHostAlloc(&l->output, l->batch*l->outputs * sizeof(float), cudaHostRegisterMapped)) {
            cudaGetLastError(); // reset CUDA-error
            l->output = (float*)xcalloc(l->batch * l->outputs, sizeof(float));
            l->output_pinned = 0;
        }
    }

    if (l->delta_pinned) {
        CHECK_CUDA(cudaFreeHost(l->delta));
        if (cudaSuccess != cudaHostAlloc(&l->delta, l->batch*l->outputs * sizeof(float), cudaHostRegisterMapped)) {
            cudaGetLastError(); // reset CUDA-error
            l->delta = (float*)xcalloc(l->batch * l->outputs, sizeof(float));
            l->delta_pinned = 0;
        }
    }



    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);
    cudaError_t status = cudaFree(l->indexes_gpu);
    check_error(status);
    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
    l->indexes_gpu =  cuda_make_int_array_ctdet(l->indexes, l->batch*l->outputs);
#endif

}

box get_ctdet_box(float *x, int index, int i, int j, int lw, int lh ,int stride)
{
    box b;
    b.x = (x[index + 0*stride]+i) / lw;
    b.y = (x[index + 1*stride]+j) / lh;
    b.w = exp(x[index + 2*stride]) / lw;
    b.h = exp(x[index + 3*stride]) / lh;
    return b;
}

float smoothL1_Loss(float x, float* x_diff){
    float loss = 0.;
    float fabs_x_value = fabs(x);
    if(fabs_x_value < 1){
        loss = 0.5 * x * x;
        *x_diff = x;
    }else{
        loss = fabs_x_value - 0.5;
        *x_diff = (0. < x) - (x < 0.);
    }
    return loss;
}

float delta_ctdet_box(box truth, float *x, int index, int i, int j, int lw, int lh, float *delta, int stride, float *box_loss)
{
    box pred = get_ctdet_box(x, index, i, j, lw, lh,stride);
    float iou = box_iou(pred, truth);

    float tx = truth.x*lw - i;
    float ty = truth.y*lh - j;
    float tw = log(truth.w*lw);
    float th = log(truth.h*lh);
    float temp_loss = 0;
    temp_loss += smoothL1_Loss(x[index + 0*stride] - tx, &delta[index + 0*stride]);
    temp_loss += smoothL1_Loss(x[index + 1*stride] - ty, &delta[index + 1*stride]);
    temp_loss += smoothL1_Loss(x[index + 2*stride] - tw, &delta[index + 2*stride]);
    temp_loss += smoothL1_Loss(x[index + 3*stride] - th, &delta[index + 3*stride]);
    *box_loss = temp_loss;
    return iou;
}


static int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(4+l.classes) + entry*l.w*l.h + loc;
}

void forward_ctdet_layer(const layer l, network_state state)
{
    int i,j,b,cl;
    memcpy(l.output, state.input, l.outputs*l.batch*sizeof(float));
    #ifndef GPU
    for (b = 0; b < l.batch; ++b){
        int index = entry_index(l, b, 0*l.w*l.h, 4);
        activate_array(l.output + index, (l.classes)*l.w*l.h, LOGISTIC);
    }
    #endif
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if(!state.train){
        return;
    }
    float avg_iou = 0;
    float recall = 0;
    float recall75 = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    *(l.cost) = 0;
    float alpha = 2., gamma_ = 4.;
    float class_cost=0, box_cost = 0;
    for (b = 0; b < l.batch; ++b) {
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i){
                for(cl = 0;cl <l.classes;++cl) {
                    int obj_index = entry_index(l, b, j * l.w + i,4 + cl);
                    float label=state.truth[obj_index];
                    float prob_obj = l.output[obj_index];
                    if(label < 1) {
                        class_cost -= pow(1 - label, gamma_) * pow(prob_obj, alpha) * log(1 - prob_obj);
                        l.delta[obj_index] = pow(prob_obj, alpha) * pow(1 - label, gamma_) *
                                                 (prob_obj - alpha * (1 - prob_obj) * log(1 - prob_obj));
                        avg_anyobj += prob_obj;

                    }else if (label == 1){
                        class_cost -= pow(1 - prob_obj, alpha) * log(prob_obj);
                        l.delta[obj_index] = pow(1 - prob_obj, alpha) * (alpha * prob_obj* log(prob_obj) - 
                                                    (1 - prob_obj));
                        int box_index = entry_index(l, b, j * l.w + i, 0);
                        box truth = float_to_box_stride(state.truth +box_index, l.w*l.h);
                        float loss = 0;
                        float iou = delta_ctdet_box(truth, l.output, box_index, i, j, l.w, l.h, l.delta, l.w*l.h, &loss);
                        avg_obj += prob_obj;
                        avg_iou+=iou;
                        box_cost += loss;
                        if(iou > .5) recall += 1;
                        if(iou > .75) recall75 += 1;
                        ++count;
                    }
                }
            }
        }
    }
    *(l.cost) = (box_cost + class_cost);
    printf("Region %d Avg IOU: %f, Obj: %f, No Obj: %f, count: %d\n", 
                        state.index, avg_iou/count, avg_obj/count, avg_anyobj/(l.classes*l.w*l.h*l.batch), count);
}

void backward_ctdet_layer(const layer l, network_state state)
{
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
}

void correct_ctdet_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw);
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth);
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

int ctdet_num_detections(layer l, float thresh)
{
    int i;
    int count = 0;
    for (i = 0; i < *(l.num_detection); ++i){
        if (l.output[l.indexes[i]] >= thresh) {
            ++count;
        }
    }
    return count;
}

int get_ctdet_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets)
{
    int i,id,obj_index,x,y,k;
    float *predictions = l.output;
    int count = 0;
    for (i = 0; i < *(l.num_detection); ++i){
        obj_index = l.indexes[i];
        id = obj_index;
        x = id % l.w;
        id /= l.w;
        y = id % l.h;
        id /= l.h;
        k = id % l.classes;
        float objectness = predictions[obj_index];
        if (objectness <= thresh) continue;
        int box_index = entry_index(l, 0, y * l.w + x, l.classes );
        dets[count].bbox = get_ctdet_box(predictions, box_index, x, y, l.w, l.h, l.w * l.h);
        dets[count].objectness = objectness;
        dets[count].classes = l.classes;
        memset(dets[count].prob,0,l.classes* sizeof(float));
        dets[count].prob[k]=objectness;
        ++count;
    }
    correct_ctdet_boxes(dets, count, w, h, netw, neth, relative);
    return count;
}

#ifdef GPU

void forward_ctdet_layer_gpu(const layer l, network_state state)
{
    simple_copy_ongpu(l.batch*l.inputs, state.input, l.output_gpu);
    for (int b = 0; b < l.batch; ++b){
        int index = entry_index(l, b, 0, 4);
        activate_array_ongpu(l.output_gpu + index, l.classes*l.w*l.h, LOGISTIC);
    }
    if(!state.train || l.onlyforward){
        cuda_pull_int_array(l.num_detection_gpu,l.num_detection,1);
        cuda_pull_int_array(l.indexes_gpu,l.indexes,*l.num_detection);
        cuda_pull_array_async(l.output_gpu, l.output, l.batch*l.outputs);
        CHECK_CUDA(cudaPeekAtLastError());
        return;
    }
    float *in_cpu = (float *)xcalloc(l.batch*l.inputs, sizeof(float));
    cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
    memcpy(in_cpu, l.output, l.batch*l.outputs*sizeof(float));
    float *truth_cpu = 0;
    
    if (state.truth) {
        int num_truth = l.batch*l.truths;
        truth_cpu = (float *)xcalloc(num_truth, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, num_truth);
    }
    network_state cpu_state = state;
    cpu_state.net = state.net;
    cpu_state.index = state.index;
    cpu_state.train = state.train;
    cpu_state.truth = truth_cpu;
    cpu_state.input = in_cpu;
    forward_ctdet_layer(l, cpu_state);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
    free(in_cpu);
    if (cpu_state.truth) free(cpu_state.truth);
}

void backward_ctdet_layer_gpu(const layer l, network_state state)
{
    axpy_ongpu(l.batch*l.inputs, 1, l.delta_gpu, 1, state.delta, 1);
}
#endif
