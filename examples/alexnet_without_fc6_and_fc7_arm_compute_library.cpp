//
// Created by zhoupeilin on 17-7-6.
//
// Because the model need to be tested by android so that
// it cannot be too big. As a result, I delete fullconnectlayer6
// and fullconnectlayer7 in this alexnet.
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/CL/CLFunctions.h"

#include "arm_compute/core/Types.h"
#include "test_helpers/Utils.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <ostream>
#include <sys/time.h>
#include <map>

using namespace arm_compute;
using namespace test_helpers;

static float StringToFloat(const std::string & str){
    std::istringstream iss(str);
    float number;
    iss >> number;
    return number;
}

void main_alexnet(int argc, const char **argv)
{
    /*----------------------------------[init_model_vgg16]-----------------------------------*/

    /*----------------------------------BEGIN:[init_Tensor]----------------------------------*/
    //init_input_tensor
    Tensor input;

    //init_conv_1_tensor
    Tensor weights_1;
    Tensor biases_1;
    Tensor out_1;
    Tensor act_1;
    Tensor pool_1;
    Tensor lrn_1;

    //init_conv_2_tensor
    Tensor weights_2;
    Tensor biases_2;
    Tensor out_2;
    Tensor act_2;
    Tensor pool_2;
    Tensor lrn_2;

    //init_conv_3_tensor
    Tensor weights_3;
    Tensor biases_3;
    Tensor out_3;
    Tensor act_3;

    //init_conv_4_tensor
    Tensor weights_4;
    Tensor biases_4;
    Tensor out_4;
    Tensor act_4;

    //init_conv_5_tensor
    Tensor weights_5;
    Tensor biases_5;
    Tensor out_5;
    Tensor act_5;
    Tensor pool_5;

    //init_fc_8
    Tensor weights_8;
    Tensor biases_8;
    Tensor out_8;

    Tensor softmax_tensor;

    //init_tensor
    constexpr unsigned int input_width  = 227;
    constexpr unsigned int input_height = 227;
    constexpr unsigned int input_fm     = 3;

    const TensorShape input_shape(input_width, input_height, input_fm);
    input.allocator() -> init(TensorInfo(input_shape, 1, DataType::F32));

    //init_conv_1
    constexpr unsigned int conv_1_kernel_x = 11;
    constexpr unsigned int conv_1_kernel_y = 11;
    constexpr unsigned int conv_1_fm       = 96;
    constexpr unsigned int conv_1_out      = 55;

    const TensorShape conv_1_weights_shape(conv_1_kernel_x, conv_1_kernel_y, input_shape.z(), conv_1_fm);
    const TensorShape conv_1_biases_shape(conv_1_weights_shape[3]);
    const TensorShape conv_1_out_shape(conv_1_out, conv_1_out, conv_1_weights_shape[3]);

    weights_1.allocator() -> init(TensorInfo(conv_1_weights_shape, 1, DataType::F32));
    biases_1.allocator() -> init(TensorInfo(conv_1_biases_shape, 1, DataType::F32));
    out_1.allocator() -> init(TensorInfo(conv_1_out_shape, 1, DataType::F32));

    act_1.allocator() -> init(TensorInfo(conv_1_out_shape, 1, DataType::F32));

    TensorShape conv_1_pool = conv_1_out_shape;
    conv_1_pool.set(0, conv_1_pool.x() / 2);
    conv_1_pool.set(1, conv_1_pool.y() / 2);
    pool_1.allocator() -> init(TensorInfo(conv_1_pool, 1, DataType::F32));

    TensorShape conv_1_lrn = conv_1_pool;
    lrn_1.allocator() -> init(TensorInfo(conv_1_lrn, 1, DataType::F32));

    //init_conv_2
    constexpr unsigned int conv_2_kernel_x = 5;
    constexpr unsigned int conv_2_kernel_y = 5;
    constexpr unsigned int conv_2_fm       = 256;

    const TensorShape conv_2_weights_shape(conv_2_kernel_x, conv_2_kernel_y, conv_1_lrn.z(), conv_2_fm);
    const TensorShape conv_2_biases_shape(conv_2_weights_shape[3]);
    const TensorShape conv_2_out_shape(conv_1_lrn.x(), conv_1_lrn.y(), conv_2_weights_shape[3]);

    weights_2.allocator() -> init(TensorInfo(conv_2_weights_shape, 1, DataType::F32));
    biases_2.allocator() -> init(TensorInfo(conv_2_biases_shape, 1, DataType::F32));
    out_2.allocator() -> init(TensorInfo(conv_2_out_shape, 1, DataType::F32));

    act_2.allocator() -> init(TensorInfo(conv_2_out_shape, 1, DataType::F32));

    TensorShape conv_2_pool = conv_2_out_shape;
    conv_2_pool.set(0, conv_2_pool.x() / 2);
    conv_2_pool.set(1, conv_2_pool.y() / 2);
    pool_2.allocator() -> init(TensorInfo(conv_2_pool, 1, DataType::F32));

    TensorShape conv_2_lrn = conv_2_pool;
    lrn_2.allocator() -> init(TensorInfo(conv_2_lrn, 1, DataType::F32));

    //init_conv_3
    constexpr unsigned int conv_3_kernel_x = 3;
    constexpr unsigned int conv_3_kernel_y = 3;
    constexpr unsigned int conv_3_fm       = 384;

    const TensorShape conv_3_weights_shape(conv_3_kernel_x, conv_3_kernel_y, conv_2_lrn.z(), conv_3_fm);
    const TensorShape conv_3_biases_shape(conv_3_weights_shape[3]);
    const TensorShape conv_3_out_shape(conv_2_lrn.x(), conv_2_lrn.y(), conv_3_weights_shape[3]);

    weights_3.allocator() -> init(TensorInfo(conv_3_weights_shape, 1, DataType::F32));
    biases_3.allocator() -> init(TensorInfo(conv_3_biases_shape, 1, DataType::F32));
    out_3.allocator() -> init(TensorInfo(conv_3_out_shape, 1, DataType::F32));

    act_3.allocator() -> init(TensorInfo(conv_3_out_shape, 1, DataType::F32));

    //init_conv_4
    constexpr unsigned int conv_4_kernel_x = 3;
    constexpr unsigned int conv_4_kernel_y = 3;
    constexpr unsigned int conv_4_fm       = 384;

    const TensorShape conv_4_weights_shape(conv_4_kernel_x, conv_4_kernel_y, conv_3_out_shape.z(), conv_4_fm);
    const TensorShape conv_4_biases_shape(conv_4_weights_shape[3]);
    const TensorShape conv_4_out_shape(conv_3_out_shape.x(), conv_3_out_shape.y(), conv_4_weights_shape[3]);

    weights_4.allocator() -> init(TensorInfo(conv_4_weights_shape, 1, DataType::F32));
    biases_4.allocator() -> init(TensorInfo(conv_4_biases_shape, 1, DataType::F32));
    out_4.allocator() -> init(TensorInfo(conv_4_out_shape, 1, DataType::F32));

    act_4.allocator() -> init(TensorInfo(conv_4_out_shape, 1, DataType::F32));

    //init_conv_5
    constexpr unsigned int conv_5_kernel_x = 3;
    constexpr unsigned int conv_5_kernel_y = 3;
    constexpr unsigned int conv_5_fm       = 256;

    const TensorShape conv_5_weights_shape(conv_5_kernel_x, conv_5_kernel_y, conv_4_out_shape.z(), conv_5_fm);
    const TensorShape conv_5_biases_shape(conv_5_weights_shape[3]);
    const TensorShape conv_5_out_shape(conv_4_out_shape.x(), conv_4_out_shape.y(), conv_5_weights_shape[3]);

    weights_5.allocator() -> init(TensorInfo(conv_5_weights_shape, 1, DataType::F32));
    biases_5.allocator() -> init(TensorInfo(conv_5_biases_shape, 1, DataType::F32));
    out_5.allocator() -> init(TensorInfo(conv_5_out_shape, 1, DataType::F32));

    act_5.allocator() -> init(TensorInfo(conv_5_out_shape, 1, DataType::F32));

    TensorShape conv_5_pool = conv_5_out_shape;
    conv_5_pool.set(0, conv_5_pool.x() / 2);
    conv_5_pool.set(1, conv_5_pool.y() / 2);
    pool_5.allocator() -> init(TensorInfo(conv_5_pool, 1, DataType::F32));

    //init_fc_8
    constexpr unsigned int fc_8_numoflabel = 100;

    const TensorShape fc_8_weights_shape(conv_5_pool.x() * conv_5_pool.y() * conv_5_pool.z(), fc_8_numoflabel);
    const TensorShape fc_8_biases_shape(fc_8_numoflabel);
    const TensorShape fc_8_out_shape(fc_8_numoflabel);

    weights_8.allocator() -> init(TensorInfo(fc_8_weights_shape, 1, DataType::F32));
    biases_8.allocator() -> init(TensorInfo(fc_8_biases_shape, 1, DataType::F32));
    out_8.allocator() -> init(TensorInfo(fc_8_out_shape, 1, DataType::F32));

    const TensorShape softmax_shape(fc_8_out_shape.x());
    softmax_tensor.allocator() -> init(TensorInfo(softmax_shape, 1, DataType::F32));

    /*----------------------------------END:[init_Tensor]----------------------------------*/


    /*-----------------------------BEGIN:[Configure Functions]-----------------------------*/
    //init_layer
    //NEON CPU
    NEConvolutionLayer    conv_1;
    NEConvolutionLayer    conv_2;
    NEConvolutionLayer    conv_3;
    NEConvolutionLayer    conv_4;
    NEConvolutionLayer    conv_5;
    NEActivationLayer     Nact_1;
    NEActivationLayer     Nact_2;
    NEActivationLayer     Nact_3;
    NEActivationLayer     Nact_4;
    NEActivationLayer     Nact_5;
    NEPoolingLayer        Npool_1;
    NEPoolingLayer        Npool_2;
    NEPoolingLayer        Npool_5;
    NENormalizationLayer  LRN_1;
    NENormalizationLayer  LRN_2;
    NEFullyConnectedLayer fc_8;
    NESoftmaxLayer        softmax;
std::cout<<"tt"<<std::endl;
    //conv_1
    //in: 227 * 227 * 3, kernel: 11 * 11 * 3 * 96, out: 55 * 55 * 96
    conv_1.configure(&input, &weights_1, &biases_1, &out_1, PadStrideInfo(4, 4, 0, 0));
    std::cout<<"tt"<<std::endl;
    //in: 55 * 55 * 96, out: 55 * 55 * 96
    Nact_1.configure(&out_1, &act_1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    //in: 55 * 55 * 96, out: 27 * 27 * 96
    Npool_1.configure(&act_1, &pool_1, PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2)));

    //in: 55 * 55 * 96, out: 27 * 27 * 96
    LRN_1.configure(&pool_1, &lrn_1, NormalizationLayerInfo(NormType::IN_MAP));

    //conv_2
    //in: 27 * 27 * 96, kernel: 5 * 5 * 96 * 256. out: 27 * 27 * 256
    conv_2.configure(&lrn_1, &weights_2, &biases_2, &out_2, PadStrideInfo(1, 1, 2, 2));

    //in: 27 * 27 * 256, out: 27 * 27 * 256
    Nact_2.configure(&out_2, &act_2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    //in: 27 * 27 * 256, out: 13 * 13 * 256
    Npool_2.configure(&act_2, &pool_2, PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2)));

    //in: 13 * 13 * 256, out: 13 * 13 * 256
    LRN_2.configure(&pool_2, &lrn_2, NormalizationLayerInfo(NormType::IN_MAP));

    //conv_3
    //in: 13 * 13 * 256, kernel: 3 * 3 * 256 * 384, out: 13 * 13 * 384
    conv_3.configure(&lrn_2, &weights_3, &biases_3, &out_3, PadStrideInfo(1, 1, 1, 1));

    //in: 13 * 13 * 384, out: 13 * 13 * 384
    Nact_3.configure(&out_3, &act_3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    //conv_4
    //in: 13 * 13 * 384, kernel: 3 * 3 * 384 * 384, out: 13 * 13 * 384
    conv_4.configure(&act_3, &weights_4, &biases_4, &out_4, PadStrideInfo(1, 1, 1, 1));

    //in: 13 * 13 * 384, out: 13 * 13 * 384
    Nact_4.configure(&out_4, &act_4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    //conv_5
    //in: 13 * 13 * 384, kernel: 13 * 13 * 384 * 256. out: 13 * 13 * 256
    conv_5.configure(&act_4, &weights_5, &biases_5, &out_5, PadStrideInfo(1, 1, 1, 1));

    //in: 13 * 13 * 256, out: 13 * 13 * 256
    Nact_5.configure(&out_5, &act_5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    //in: 13 * 13 * 256, out: 6 * 6 * 256
    Npool_5.configure(&act_5, &pool_5, PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2)));

    //fc_8
    //in: 6 * 6 * 256, out: 1000
    fc_8.configure(&pool_5, &weights_8, &biases_8, &out_8);

    //softmax layer: 1000
    softmax.configure(&out_8, &softmax_tensor);

    /*------------------------------END:[Configure Functions]------------------------------*/

    /*------------------------------BEGIN:[Allocate tensors]-------------------------------*/

    //input
    input.allocator() -> allocate();

    //conv_1
    weights_1.allocator() -> allocate();
    biases_1.allocator() -> allocate();
    out_1.allocator() -> allocate();
    act_1.allocator() -> allocate();
    pool_1.allocator() -> allocate();
    lrn_1.allocator() -> allocate();

    //conv_2
    weights_2.allocator() -> allocate();
    biases_2.allocator() -> allocate();
    out_2.allocator() -> allocate();
    act_2.allocator() -> allocate();
    pool_2.allocator() -> allocate();
    lrn_2.allocator() -> allocate();

    //conv_3
    weights_3.allocator() -> allocate();
    biases_3.allocator() -> allocate();
    out_3.allocator() -> allocate();
    act_3.allocator() -> allocate();

    //conv_4
    weights_4.allocator() -> allocate();
    biases_4.allocator() -> allocate();
    out_4.allocator() -> allocate();
    act_4.allocator() -> allocate();

    //conv_5
    weights_5.allocator() -> allocate();
    biases_5.allocator() -> allocate();
    out_5.allocator() -> allocate();
    act_5.allocator() -> allocate();
    pool_5.allocator() -> allocate();

    //fc_8
    weights_8.allocator() -> allocate();
    biases_8.allocator() -> allocate();
    out_8.allocator() -> allocate();
    softmax_tensor.allocator() -> allocate();

    /*------------------------------END:[Allocate tensors]-------------------------------*/

    /*-----------------------------BEGIN:[Load the weights]------------------------------*/

    /*------------------------------END:[Load the weights]-------------------------------*/

    /*-----------------------------------BEGIN:[Input]-----------------------------------*/

    /*------------------------------------END:[Input]------------------------------------*/

    /*--------------------------BEGIN:[Execute the functions]----------------------------*/

    //time
    struct timeval start, end;
    gettimeofday(&start, NULL);

    //conv_1
    conv_1.run();
    Nact_1.run();
    conv_1.run();
    Nact_1.run();
    Npool_1.run();
    LRN_1.run();

    //conv_2
    conv_2.run();
    Nact_2.run();
    conv_2.run();
    Nact_2.run();
    Npool_2.run();
    LRN_2.run();

    //conv_3
    conv_3.run();
    Nact_3.run();

    //conv_4
    conv_4.run();
    Nact_4.run();

    //conv_5
    conv_5.run();
    Nact_5.run();
    conv_5.run();
    Nact_5.run();
    Npool_5.run();

    //fc_8
    fc_8.run();
    softmax.run();

    gettimeofday(&end, NULL);
    /*---------------------------END:[Execute the functions]-----------------------------*/

    //test

    std::cout << std::endl << std::endl << std::endl;
    int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec -start.tv_usec;
    printf("time: %d us\n", timeuse);

    std::cout << "fine! Jason!" << std::endl;
}


/** Main program for convolution test
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Path to PPM image to process )
 */
int main(int argc, const char **argv)
{
    return test_helpers::run_example(argc, argv, main_alexnet);
}