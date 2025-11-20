#ifndef _HAND_LANDMARKS_H_
#define _HAND_LANDMARKS_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "rknn_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Hand landmark structure (21 landmarks per hand)
typedef struct {
    float x;
    float y;
    float z;  // depth
    float score;  // confidence score
} hand_landmark_point_t;

// Hand detection result - SOLO C puro
typedef struct {
    float confidence;
    int hand_id;
} hand_detect_result_t;

// Results list for multiple hands
typedef struct {
    int count;
} hand_detect_result_list;

// RKNN context and model information
typedef struct {
    rknn_context ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    rknn_tensor_mem* input_mems;      // CORRECT: pointer returned by rknn_create_mem
    rknn_tensor_mem** output_mems;    // CORRECT: array of POINTERS
    int input_index;
    int output_index;
    
    // Model parameters
    int model_width;
    int model_height;
    int model_channel;
} rknn_app_context_t;

/**
 * @brief Initialize hand landmarks detection model
 */
int init_hand_landmarks_model(const char *model_path, rknn_app_context_t *app_ctx);

/**
 * @brief Run hand landmarks inference
 */
int inference_hand_landmarks_model(rknn_app_context_t *app_ctx, 
                                  float* landmarks_out,  // Output: 21*3 = 63 floats
                                  int* num_landmarks);

/**
 * @brief Release hand landmarks detection model resources
 */
int release_hand_landmarks_model(rknn_app_context_t *app_ctx);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // _HAND_LANDMARKS_H_