#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/poll.h>
#include <time.h>
#include <unistd.h>
#include <vector>

#include "rtsp_demo.h"
#include "luckfox_mpi.h"
#include "hand_landmarks.h" // Tu header

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#define DISP_WIDTH  720
#define DISP_HEIGHT 480

// Función para dibujar manos (La misma que tenías)
void draw_hand_landmarks(cv::Mat &frame, float *landmarks, int num_landmarks,
                        int model_width, int model_height, int display_width, int display_height) {
    float scale_x = (float)display_width / (float)model_width;
    float scale_y = (float)display_height / (float)model_height;
    int connections[][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 4}, {0, 5}, {5, 6}, {6, 7}, {7, 8},
        {0, 9}, {9, 10}, {10, 11}, {11, 12}, {0, 13}, {13, 14}, {14, 15}, {15, 16},
        {0, 17}, {17, 18}, {18, 19}, {19, 20}
    };
    if (num_landmarks <= 0) return;
    for (int i = 0; i < 21; i++) {
        int x = (int)(landmarks[i * 3 + 0] * model_width * scale_x);
        int y = (int)(landmarks[i * 3 + 1] * model_height * scale_y);
        cv::circle(frame, cv::Point(x, y), 3, cv::Scalar(0, 255, 0), -1);
    }
    for (size_t i = 0; i < sizeof(connections) / sizeof(connections[0]); i++) {
        int idx1 = connections[i][0];
        int idx2 = connections[i][1];
        int x1 = (int)(landmarks[idx1 * 3 + 0] * model_width * scale_x);
        int y1 = (int)(landmarks[idx1 * 3 + 1] * model_height * scale_y);
        int x2 = (int)(landmarks[idx2 * 3 + 0] * model_width * scale_x);
        int y2 = (int)(landmarks[idx2 * 3 + 1] * model_height * scale_y);
        cv::line(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 2);
    }
}

int main(int argc, char *argv[]) {
    printf(">>> STARTING FIXED HANDLANDMARKS_RTSP V2 <<<\n");
    system("RkLunch-stop.sh");
    RK_S32 s32Ret = 0;

    // Configuración igual al ejemplo
    int width = DISP_WIDTH;
    int height = DISP_HEIGHT;

    // Inicializar Modelo RKNN
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    printf("[DEBUG] Loading hand landmarks model...\n");
    if (init_hand_landmarks_model("./model/hands.rknn", &rknn_app_ctx) < 0) {
        printf("[ERROR] Failed to init hand landmarks model\n");
        return -1;
    }
    printf("[DEBUG] Model loaded successfully!\n");
    int model_width = rknn_app_ctx.model_width;
    int model_height = rknn_app_ctx.model_height;
    printf("[DEBUG] Model dimensions: %dx%d\n", model_width, model_height);

    // Inicializar Variables de Video
    VENC_STREAM_S stFrame;
    stFrame.pstPack = (VENC_PACK_S *)malloc(sizeof(VENC_PACK_S));
    VIDEO_FRAME_INFO_S stViFrame;

    // --- ARQUITECTURA DE MEMORIA DEL EJEMPLO (CLAVE PARA QUE FUNCIONE) ---
    MB_POOL_CONFIG_S PoolCfg;
    memset(&PoolCfg, 0, sizeof(MB_POOL_CONFIG_S));
    PoolCfg.u64MBSize = width * height * 3; // Espacio para RGB/BGR
    PoolCfg.u32MBCnt = 1;
    PoolCfg.enAllocType = MB_ALLOC_TYPE_DMA; // Esto arregla el crash del encoder
    MB_POOL src_Pool = RK_MPI_MB_CreatePool(&PoolCfg);
    printf("Create Pool success!\n");

    // Obtener bloque de memoria DMA
    MB_BLK src_Blk = RK_MPI_MB_GetMB(src_Pool, width * height * 3, RK_TRUE);
    
    // Configurar frame H264 apuntando a ese bloque
    VIDEO_FRAME_INFO_S h264_frame;
    memset(&h264_frame, 0, sizeof(VIDEO_FRAME_INFO_S)); // Limpiar struct
    h264_frame.stVFrame.u32Width = width;
    h264_frame.stVFrame.u32Height = height;
    h264_frame.stVFrame.u32VirWidth = width;
    h264_frame.stVFrame.u32VirHeight = height;
    h264_frame.stVFrame.enPixelFormat = RK_FMT_RGB888; // El ejemplo usa esto (OpenCV escribe BGR aquí)
    h264_frame.stVFrame.u32FrameFlag = 160;
    h264_frame.stVFrame.pMbBlk = src_Blk; // Apuntar al bloque DMA
    
    // Mapear OpenCV a la memoria DMA
    unsigned char *data = (unsigned char *)RK_MPI_MB_Handle2VirAddr(src_Blk);
    cv::Mat frame(cv::Size(width, height), CV_8UC3, data);
    // ---------------------------------------------------------------------

    // --- BLOQUE FALTANTE: INICIAR ISP (RKAIQ) ---
    // Esto es obligatorio para que funcionen video11 y video12
    rk_aiq_working_mode_t hdr_mode = RK_AIQ_WORKING_MODE_NORMAL;
    RK_BOOL multi_sensor = RK_FALSE;
    const char *iq_dir = "/etc/iqfiles"; // Ruta estándar en Luckfox
    
    SAMPLE_COMM_ISP_Init(0, hdr_mode, multi_sensor, iq_dir);
    SAMPLE_COMM_ISP_Run(0);

    // 1. Inicializar MPI System
    if (RK_MPI_SYS_Init() != RK_SUCCESS) {
        RK_LOGE("rk mpi sys init fail!");
        return -1;
    }

    // Init RTSP
    rtsp_demo_handle g_rtsplive = NULL;
    rtsp_session_handle g_rtsp_session;
    g_rtsplive = create_rtsp_demo(554);
    g_rtsp_session = rtsp_new_session(g_rtsplive, "/live/0");
    rtsp_set_video(g_rtsp_session, RTSP_CODEC_ID_VIDEO_H264, NULL, 0);
    rtsp_sync_video_ts(g_rtsp_session, rtsp_get_reltime(), rtsp_get_ntptime());

    // Init VI (Canal 0 como en el ejemplo, ya que tenemos DMABUF configurado)
    vi_dev_init();
    vi_chn_init(0, width, height);

    // Init VENC
    if (venc_init(0, width, height, RK_VIDEO_ID_AVC) != 0) {
        printf("venc_init failed\n");
        return -1;
    }

    printf("Init Success\n");

    int frame_count = 0;
    // Buffer para inferencia
    cv::Mat model_bgr(model_height, model_width, CV_8UC3);
    float landmarks[63]; // 21 * 3
    int num_landmarks = 0;

    while(1) {
        // 1. Obtener Frame de Cámara (NV12)
        h264_frame.stVFrame.u64PTS = TEST_COMM_GetNowUs();
        // 2. Get Frame (Blocking, 30 FPS from camera)
        s32Ret = RK_MPI_VI_GetChnFrame(0, 0, &stViFrame, 1000); // 1000ms timeout
        if (s32Ret == RK_SUCCESS) {
            void *data_vi = RK_MPI_MB_Handle2VirAddr(stViFrame.stVFrame.pMbBlk);
            
            // Downsample to 15 FPS: Skip every other frame
            if (frame_count % 2 != 0) {
                RK_MPI_VI_ReleaseChnFrame(0, 0, &stViFrame);
                frame_count++;
                // printf("[DEBUG] Skipping frame %d\n", frame_count);
                continue;
            }
            // printf("[DEBUG] Processing frame %d\n", frame_count);

            // Create OpenCV mat for the YUV frame
            cv::Mat yuv420sp(height + height / 2, width, CV_8UC1, data_vi);
            
            // Convert YUV -> BGR directly to the DMA memory ('frame' points to 'data' which is DMA)
            // This fills 'h264_frame' with the BGR image ready for processing and encoding
            cv::cvtColor(yuv420sp, frame, cv::COLOR_YUV420sp2BGR);

            // 3. Inferencia - prepare input
            cv::resize(frame, model_bgr, cv::Size(model_width, model_height), 0, 0, cv::INTER_LINEAR);
            
            // Run inference on EVERY processed frame (15 FPS)
            // RV1103 uses zero-copy: copy data directly to input_mems
            memcpy(rknn_app_ctx.input_mems->virt_addr, model_bgr.data, model_width * model_height * 3);
            
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
            inference_hand_landmarks_model(&rknn_app_ctx, landmarks, &num_landmarks);
            
            clock_gettime(CLOCK_MONOTONIC, &end);
            long ms = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
            
            if (num_landmarks > 0) {
                // Only print timing occasionally to avoid spam
                if (frame_count % 30 == 0) { 
                     printf("[TIMING] Inference took %ld ms - Hand detected!\n", ms);
                }
                draw_hand_landmarks(frame, landmarks, num_landmarks, model_width, model_height, width, height);
            }
            
            // 6. Liberar frame de cámara
            RK_MPI_VI_ReleaseChnFrame(0, 0, &stViFrame);
            frame_count++;
        }
        // Send to encoder
        s32Ret = RK_MPI_VENC_SendFrame(0, &h264_frame, -1);
        if (s32Ret != RK_SUCCESS) {
            printf("[ERROR] RK_MPI_VENC_SendFrame failed: %x\n", s32Ret);
        }

        // Get encoded stream
        s32Ret = RK_MPI_VENC_GetStream(0, &stFrame, 200); // 200ms timeout
        if (s32Ret == RK_SUCCESS) {
            if (g_rtsplive && g_rtsp_session) {
                void *pData = RK_MPI_MB_Handle2VirAddr(stFrame.pstPack->pMbBlk);
                rtsp_tx_video(g_rtsp_session, (const uint8_t *)pData,
                              stFrame.pstPack->u32Len, stFrame.pstPack->u64PTS);
                rtsp_do_event(g_rtsplive);
            } else {
                printf("[ERROR] RTSP session not available\n");
            }
            RK_MPI_VENC_ReleaseStream(0, &stFrame);
        }
        
        frame_count++;
        // Sleep to achieve ~5 FPS
    }

    // Limpieza
    RK_MPI_MB_ReleaseMB(src_Blk);
    RK_MPI_MB_DestroyPool(src_Pool);
    RK_MPI_VI_DisableChn(0, 0);
    RK_MPI_VI_DisableDev(0);
    RK_MPI_VENC_StopRecvFrame(0);
    RK_MPI_VENC_DestroyChn(0);
    free(stFrame.pstPack);
    if (g_rtsplive) rtsp_del_demo(g_rtsplive);
    RK_MPI_SYS_Exit();
    
    // Release model (asumiendo que tienes esta función en tu .h)
    printf("[DEBUG] Releasing model...\n");
    release_hand_landmarks_model(&rknn_app_ctx);
    printf("[DEBUG] Model released\n");
    return 0;
}