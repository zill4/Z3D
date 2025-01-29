#include "rasterizer.h"

void rasterizeTriangleCPU(int idx, float* vt0, float* vt1, float* vt2, int width, int height, INT64* zbuffer, float* d, float occlusion_truncation) {
    float x_min = std::min(vt0[0], std::min(vt1[0],vt2[0]));
    float x_max = std::max(vt0[0], std::max(vt1[0],vt2[0]));
    float y_min = std::min(vt0[1], std::min(vt1[1],vt2[1]));
    float y_max = std::max(vt0[1], std::max(vt1[1],vt2[1]));

    for (int px = x_min; px < x_max + 1; ++px) {
        if (px < 0 || px >= width)
            continue;
        for (int py = y_min; py < y_max + 1; ++py) {
            if (py < 0 || py >= height)
                continue;
            float vt[2] = {px + 0.5, py + 0.5};
            float baryCentricCoordinate[3];
            calculateBarycentricCoordinate(vt0, vt1, vt2, vt, baryCentricCoordinate);
            if (isBarycentricCoordInBounds(baryCentricCoordinate)) {
                int pixel = py * width + px;
                if (zbuffer == 0) {
                    zbuffer[pixel] = (INT64)(idx + 1);
                    continue;
                }

                float depth = baryCentricCoordinate[0] * vt0[2] + baryCentricCoordinate[1] * vt1[2] + baryCentricCoordinate[2] * vt2[2];
                float depth_thres = 0;
                if (d) {
                    depth_thres = d[pixel] * 0.49999f + 0.5f + occlusion_truncation;
                }
                
                int z_quantize = depth * (2<<17);
                INT64 token = (INT64)z_quantize * MAXINT + (INT64)(idx + 1);
                if (depth < depth_thres)
                    continue;
                zbuffer[pixel] = std::min(zbuffer[pixel], token);
            }
        }
    }
}

void barycentricFromImgcoordCPU(float* V, int* F, int* findices, INT64* zbuffer, int width, int height, int num_vertices, int num_faces,
    float* barycentric_map, int pix)
{
    INT64 f = zbuffer[pix] % MAXINT;
    if (f == (MAXINT-1)) {
        findices[pix] = 0;
        barycentric_map[pix * 3] = 0;
        barycentric_map[pix * 3 + 1] = 0;
        barycentric_map[pix * 3 + 2] = 0;
        return;
    }
    findices[pix] = f;
    f -= 1;
    float barycentric[3] = {0, 0, 0};
    if (f >= 0) {
        float vt[2] = {float(pix % width) + 0.5f, float(pix / width) + 0.5f};
        float* vt0_ptr = V + (F[f * 3] * 4);
        float* vt1_ptr = V + (F[f * 3 + 1] * 4);
        float* vt2_ptr = V + (F[f * 3 + 2] * 4);

        float vt0[2] = {(vt0_ptr[0] / vt0_ptr[3] * 0.5f + 0.5f) * (width - 1) + 0.5f, (0.5f + 0.5f * vt0_ptr[1] / vt0_ptr[3]) * (height - 1) + 0.5f};
        float vt1[2] = {(vt1_ptr[0] / vt1_ptr[3] * 0.5f + 0.5f) * (width - 1) + 0.5f, (0.5f + 0.5f * vt1_ptr[1] / vt1_ptr[3]) * (height - 1) + 0.5f};
        float vt2[2] = {(vt2_ptr[0] / vt2_ptr[3] * 0.5f + 0.5f) * (width - 1) + 0.5f, (0.5f + 0.5f * vt2_ptr[1] / vt2_ptr[3]) * (height - 1) + 0.5f};

        calculateBarycentricCoordinate(vt0, vt1, vt2, vt, barycentric);

        barycentric[0] = barycentric[0] / vt0_ptr[3];
        barycentric[1] = barycentric[1] / vt1_ptr[3];
        barycentric[2] = barycentric[2] / vt2_ptr[3];
        float w = 1.0f / (barycentric[0] + barycentric[1] + barycentric[2]);
        barycentric[0] *= w;
        barycentric[1] *= w;
        barycentric[2] *= w;

    }
    barycentric_map[pix * 3] = barycentric[0];
    barycentric_map[pix * 3 + 1] = barycentric[1];
    barycentric_map[pix * 3 + 2] = barycentric[2];
}

void rasterizeImagecoordsKernelCPU(float* V, int* F, float* d, INT64* zbuffer, float occlusion_trunc, int width, int height, int num_vertices, int num_faces, int f)
{
    float* vt0_ptr = V + (F[f * 3] * 4);
    float* vt1_ptr = V + (F[f * 3 + 1] * 4);
    float* vt2_ptr = V + (F[f * 3 + 2] * 4);

    float vt0[3] = {(vt0_ptr[0] / vt0_ptr[3] * 0.5f + 0.5f) * (width - 1) + 0.5f, (0.5f + 0.5f * vt0_ptr[1] / vt0_ptr[3]) * (height - 1) + 0.5f, vt0_ptr[2] / vt0_ptr[3] * 0.49999f + 0.5f};
    float vt1[3] = {(vt1_ptr[0] / vt1_ptr[3] * 0.5f + 0.5f) * (width - 1) + 0.5f, (0.5f + 0.5f * vt1_ptr[1] / vt1_ptr[3]) * (height - 1) + 0.5f, vt1_ptr[2] / vt1_ptr[3] * 0.49999f + 0.5f};
    float vt2[3] = {(vt2_ptr[0] / vt2_ptr[3] * 0.5f + 0.5f) * (width - 1) + 0.5f, (0.5f + 0.5f * vt2_ptr[1] / vt2_ptr[3]) * (height - 1) + 0.5f, vt2_ptr[2] / vt2_ptr[3] * 0.49999f + 0.5f};

    rasterizeTriangleCPU(f, vt0, vt1, vt2, width, height, zbuffer, d, occlusion_trunc);
}

std::vector<torch::Tensor> rasterize_image_cpu(torch::Tensor V, torch::Tensor F, torch::Tensor D,
    int width, int height, float occlusion_truncation, int use_depth_prior)
{
    int num_faces = F.size(0);
    int num_vertices = V.size(0);
    auto options = torch::TensorOptions().dtype(torch::kInt32).requires_grad(false);
    auto INT64_options = torch::TensorOptions().dtype(torch::kInt64).requires_grad(false);
    auto findices = torch::zeros({height, width}, options);
    INT64 maxint = (INT64)MAXINT * (INT64)MAXINT + (MAXINT - 1);
    auto z_min = torch::ones({height, width}, INT64_options) * (int64_t)maxint;

    if (!use_depth_prior) {
        for (int i = 0; i < num_faces; ++i) {
            rasterizeImagecoordsKernelCPU(V.data_ptr<float>(), F.data_ptr<int>(), 0,
                (INT64*)z_min.data_ptr<int64_t>(), occlusion_truncation, width, height, num_vertices, num_faces, i); 
        }
    } else {
        for (int i = 0; i < num_faces; ++i)
            rasterizeImagecoordsKernelCPU(V.data_ptr<float>(), F.data_ptr<int>(), D.data_ptr<float>(),
                (INT64*)z_min.data_ptr<int64_t>(), occlusion_truncation, width, height, num_vertices, num_faces, i);
    }

    auto float_options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    auto barycentric = torch::zeros({height, width, 3}, float_options);
    for (int i = 0; i < width * height; ++i)
        barycentricFromImgcoordCPU(V.data_ptr<float>(), F.data_ptr<int>(),
            findices.data_ptr<int>(), (INT64*)z_min.data_ptr<int64_t>(), width, height, num_vertices, num_faces, barycentric.data_ptr<float>(), i);

    return {findices, barycentric};
}

std::vector<torch::Tensor> rasterize_image(torch::Tensor V, torch::Tensor F, torch::Tensor D,
    int width, int height, float occlusion_truncation, int use_depth_prior)
{
    int device_id = V.get_device();
    if (device_id == -1)
        return rasterize_image_cpu(V, F, D, width, height, occlusion_truncation, use_depth_prior);
    else
        return rasterize_image_gpu(V, F, D, width, height, occlusion_truncation, use_depth_prior);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_image", &rasterize_image, "Custom image rasterization");
  m.def("build_hierarchy", &build_hierarchy, "Custom image rasterization");
  m.def("build_hierarchy_with_feat", &build_hierarchy_with_feat, "Custom image rasterization");
}
