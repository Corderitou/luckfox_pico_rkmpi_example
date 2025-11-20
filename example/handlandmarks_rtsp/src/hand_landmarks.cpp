#include "hand_landmarks.h"
#include <string.h>
#include <math.h>
#include <stdlib.h> // malloc, free

#define RK_SUCCESS 0
#define RK_FAIL (-1)

// Helper function to read model from file
static unsigned char* load_model(const char *filename, int *model_size) {
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        printf("open file %s failed\n", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    unsigned char *model_data = (unsigned char *)malloc(size);
    if (model_data == NULL) {
        printf("malloc model data failed\n");
        fclose(fp);
        return NULL;
    }
    if (fread(model_data, 1, size, fp) != (size_t)size) {
        printf("read model data failed\n");
        free(model_data);
        fclose(fp);
        return NULL;
    }
    fclose(fp);

    *model_size = size;
    return model_data;
}

// Initialize model
int init_hand_landmarks_model(const char *model_path, rknn_app_context_t *app_ctx) {
    int ret = RK_SUCCESS;
    int model_size = 0;
    
    // 1. Load Model
    unsigned char *model_data = load_model(model_path, &model_size);
    if (model_data == NULL) {
        return RK_FAIL;
    }

    // 2. Init RKNN  
    // Try without flags first - SDK 1.6.0 should support rknn_inputs_set by default
    ret = rknn_init(&app_ctx->ctx, model_data, model_size, 0, NULL);
    free(model_data);
    if (ret < 0) {
        printf("[ERROR] rknn_init fail! ret=%d\n", ret);
        return RK_FAIL;
    }
    printf("[DEBUG] rknn_init success, context created\n");

    // 3. Get SDK Version (Optional but good for debug)
    rknn_sdk_version version;
    ret = rknn_query(app_ctx->ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret == RK_SUCCESS) {
        printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);
    }

    // 4. Get Model Input/Output Numbers
    ret = rknn_query(app_ctx->ctx, RKNN_QUERY_IN_OUT_NUM, &app_ctx->io_num, sizeof(rknn_input_output_num));
    if (ret < 0) {
        printf("rknn_query io_num fail! ret=%d\n", ret);
        return RK_FAIL;
    }

    // 5. Get Input Attributes
    app_ctx->input_attrs = (rknn_tensor_attr *)malloc(app_ctx->io_num.n_input * sizeof(rknn_tensor_attr));
    memset(app_ctx->input_attrs, 0, app_ctx->io_num.n_input * sizeof(rknn_tensor_attr));
    for (int i = 0; i < app_ctx->io_num.n_input; i++) {
        app_ctx->input_attrs[i].index = i;
        ret = rknn_query(app_ctx->ctx, RKNN_QUERY_INPUT_ATTR, &(app_ctx->input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("rknn_query input_attr[%d] fail! ret=%d\n", i, ret);
            return RK_FAIL;
        }
    }
    
    // CRITICAL: Re-query input attrs to ensure context accepts rknn_inputs_set
    // This is a workaround for SDK 1.6.0
    for (int i = 0; i < app_ctx->io_num.n_input; i++) {
        app_ctx->input_attrs[i].index = i;
        ret = rknn_query(app_ctx->ctx, RKNN_QUERY_INPUT_ATTR, &(app_ctx->input_attrs[i]), sizeof(rknn_tensor_attr));
    }

    // 6. Get Output Attributes
    app_ctx->output_attrs = (rknn_tensor_attr *)malloc(app_ctx->io_num.n_output * sizeof(rknn_tensor_attr));
    memset(app_ctx->output_attrs, 0, app_ctx->io_num.n_output * sizeof(rknn_tensor_attr));
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        app_ctx->output_attrs[i].index = i;
        ret = rknn_query(app_ctx->ctx, RKNN_QUERY_OUTPUT_ATTR, &(app_ctx->output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("rknn_query output_attr[%d] fail! ret=%d\n", i, ret);
            return RK_FAIL;
        }
    }

    // Save model parameters for main.cc
    if (app_ctx->input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        app_ctx->model_channel = app_ctx->input_attrs[0].dims[1];
        app_ctx->model_height  = app_ctx->input_attrs[0].dims[2];
        app_ctx->model_width   = app_ctx->input_attrs[0].dims[3];
    } else {
        app_ctx->model_height  = app_ctx->input_attrs[0].dims[1];
        app_ctx->model_width   = app_ctx->input_attrs[0].dims[2];
        app_ctx->model_channel = app_ctx->input_attrs[0].dims[3];
    }
    
    printf("\n========== RKNN MODEL INFO ==========\n");
    printf("Model input: w=%d, h=%d, c=%d\n", app_ctx->model_width, app_ctx->model_height, app_ctx->model_channel);
    
    // Print all input details
    printf("\n--- INPUTS (%d total) ---\n", app_ctx->io_num.n_input);
    for (int i = 0; i < app_ctx->io_num.n_input; i++) {
        printf("Input %d:\n", i);
        printf("  Name: %s\n", app_ctx->input_attrs[i].name);
        printf("  Index: %d\n", app_ctx->input_attrs[i].index);
        printf("  Format: %s\n", app_ctx->input_attrs[i].fmt == RKNN_TENSOR_NCHW ? "NCHW" : "NHWC");
        printf("  Type: %d (1=FP32, 2=FP16, 3=INT8, 4=UINT8, 5=INT16)\n", app_ctx->input_attrs[i].type);
        printf("  Qnt Type: %d (0=NONE, 1=DFP, 2=AFFINE_ASYMMETRIC)\n", app_ctx->input_attrs[i].qnt_type);
        if (app_ctx->input_attrs[i].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC) {
            printf("  Quantization: scale=%.6f, zp=%d\n", app_ctx->input_attrs[i].scale, app_ctx->input_attrs[i].zp);
        }
        printf("  Dims: [%d, %d, %d, %d]\n", 
               app_ctx->input_attrs[i].dims[0], app_ctx->input_attrs[i].dims[1],
               app_ctx->input_attrs[i].dims[2], app_ctx->input_attrs[i].dims[3]);
        printf("  Size: %d bytes\n", app_ctx->input_attrs[i].size);
        printf("  Size with stride: %d bytes\n", app_ctx->input_attrs[i].size_with_stride);
    }
    
    // Print all output details
    printf("\n--- OUTPUTS (%d total) ---\n", app_ctx->io_num.n_output);
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        printf("Output %d:\n", i);
        printf("  Name: %s\n", app_ctx->output_attrs[i].name);
        printf("  Index: %d\n", app_ctx->output_attrs[i].index);
        printf("  Format: %s\n", app_ctx->output_attrs[i].fmt == RKNN_TENSOR_NCHW ? "NCHW" : "NHWC");
        printf("  Type: %d (1=FP32, 2=FP16, 3=INT8, 4=UINT8, 5=INT16)\n", app_ctx->output_attrs[i].type);
        printf("  Qnt Type: %d (0=NONE, 1=DFP, 2=AFFINE_ASYMMETRIC)\n", app_ctx->output_attrs[i].qnt_type);
        if (app_ctx->output_attrs[i].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC) {
            printf("  Quantization: scale=%.6f, zp=%d\n", app_ctx->output_attrs[i].scale, app_ctx->output_attrs[i].zp);
        }
        printf("  Dims: [%d, %d, %d, %d]\n", 
               app_ctx->output_attrs[i].dims[0], app_ctx->output_attrs[i].dims[1],
               app_ctx->output_attrs[i].dims[2], app_ctx->output_attrs[i].dims[3]);
        printf("  Size: %d bytes\n", app_ctx->output_attrs[i].size);
        printf("  Size with stride: %d bytes\n", app_ctx->output_attrs[i].size_with_stride);
    }
    printf("=====================================\n\n");

    // NOTE: RV1103 does NOT support rknn_inputs_set/rknn_outputs_get
    // MUST use zero-copy method with rknn_set_io_mem
    printf("[INFO] Using zero-copy method (rknn_set_io_mem) - required for RV1103\n");
    
    // 7. Create Memory for Input (Zero Copy)
    app_ctx->input_mems = rknn_create_mem(app_ctx->ctx, app_ctx->input_attrs[0].size_with_stride);
    if (app_ctx->input_mems == NULL) {
        printf("[ERROR] rknn_create_mem input fail\n");
        return RK_FAIL;
    }
    
    // Set input memory properties  
    ret = rknn_set_io_mem(app_ctx->ctx, app_ctx->input_mems, &app_ctx->input_attrs[0]);
    if (ret < 0) {
        printf("[ERROR] rknn_set_io_mem input fail! ret=%d\n", ret);
        return RK_FAIL;
    }
    printf("[DEBUG] Input memory configured for zero-copy\n");

    // 8. Create Memory for Outputs
    app_ctx->output_mems = (rknn_tensor_mem **)malloc(app_ctx->io_num.n_output * sizeof(rknn_tensor_mem*));
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        app_ctx->output_mems[i] = rknn_create_mem(app_ctx->ctx, app_ctx->output_attrs[i].size_with_stride);
        if (app_ctx->output_mems[i] == NULL) {
             printf("[ERROR] rknn_create_mem output[%d] fail\n", i);
             return RK_FAIL;
        }
        ret = rknn_set_io_mem(app_ctx->ctx, app_ctx->output_mems[i], &app_ctx->output_attrs[i]);
        if (ret < 0) {
            printf("[ERROR] rknn_set_io_mem output[%d] fail! ret=%d\n", i, ret);
            return RK_FAIL;
        }
    }
    printf("[DEBUG] Output memory configured for zero-copy\n");

    return RK_SUCCESS;
}

// Inference Function (Zero-Copy for RV1103)
int inference_hand_landmarks_model(rknn_app_context_t *app_ctx, float *landmarks_out, int *num_landmarks) {
    int ret;

    // Input is already in app_ctx->input_mems->virt_addr (copied in main.cc)
    // Just run inference
    ret = rknn_run(app_ctx->ctx, NULL);
    if (ret < 0) {
        printf("[ERROR] rknn_run fail! ret=%d\n", ret);
        return RK_FAIL;
    }

    // Outputs are in output_mems - read directly from there
    // Output 0 should contain the landmarks (63 bytes FP16 quantized)
    if (app_ctx->io_num.n_output > 0 && app_ctx->output_mems[0] != NULL) {
        // The output is FP16 quantized, need to de-quantize manually
        // Using formula: float_value = (quantized_value - zp) * scale
        int8_t *output_data = (int8_t *)app_ctx->output_mems[0]->virt_addr;
        float scale = app_ctx->output_attrs[0].scale;
        int zp = app_ctx->output_attrs[0].zp;
        
        if (output_data != NULL) {
            // Debug prints commented out for performance
            /*
            printf("[DEBUG] Output raw bytes (first 10): ");
            for (int i = 0; i < 10 && i < app_ctx->output_attrs[0].size; i++) {
                printf("%d ", output_data[i]);
            }
            printf("\n");
            */
            
            // De-quantize 63 values (21 landmarks * 3 coords)
            // Model outputs coordinates in pixel space [0, 224], need to normalize to [0, 1]
            bool valid = true;
            int invalid_count = 0;
            for (int i = 0; i < 63 && i < app_ctx->output_attrs[0].size; i++) {
                float dequantized = (output_data[i] - zp) * scale;
                // Normalize to [0, 1] by dividing by model input size
                float normalized = dequantized / (float)app_ctx->model_width;
                landmarks_out[i] = normalized;
                
                // Sanity check - normalized coords should be in [0, 1] (allow slight overflow)
                if (isnan(normalized) || isinf(normalized) || normalized < -0.5f || normalized > 1.5f) {
                    invalid_count++;
                    if (invalid_count > 10) {  // Allow some outliers
                        valid = false;
                    }
                }
            }
            
            // Debug prints commented out for performance
            /*
            printf("[DEBUG] First 6 normalized values: ");
            for (int i = 0; i < 6; i++) {
                printf("%.3f ", landmarks_out[i]);
            }
            printf("(invalid_count=%d, valid=%d)\n", invalid_count, valid);
            */
            
            if (valid) {
                *num_landmarks = 21;
                //printf("[DEBUG] Hand detected! Setting num_landmarks=21\n");
            } else {
                *num_landmarks = 0;
                //printf("[DEBUG] Invalid data, num_landmarks=0\n");
            }
        } else {
            *num_landmarks = 0;
        }
    } else {
        *num_landmarks = 0;
    }

    return RK_SUCCESS;
}

// Release
int release_hand_landmarks_model(rknn_app_context_t *app_ctx) {
    if (app_ctx->input_mems) {
        rknn_destroy_mem(app_ctx->ctx, app_ctx->input_mems);
    }
    if (app_ctx->output_mems) {
        for (int i = 0; i < app_ctx->io_num.n_output; i++) {
            if (app_ctx->output_mems[i]) {
                rknn_destroy_mem(app_ctx->ctx, app_ctx->output_mems[i]);
            }
        }
        free(app_ctx->output_mems);
    }
    if (app_ctx->input_attrs) free(app_ctx->input_attrs);
    if (app_ctx->output_attrs) free(app_ctx->output_attrs);
    if (app_ctx->ctx) rknn_destroy(app_ctx->ctx);
    return 0;
}