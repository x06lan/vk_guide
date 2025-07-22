//> includes
#include "vk_engine.h"
#include "vk_images.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/packing.hpp>
#include <vk_initializers.h>
#include <vk_types.h>

#include "VkBootstrap.h"
#include "vk_pipelines.h"
#include <chrono>
#include <thread>
#define VMA_IMPLEMENTATION
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"
#include <vk_mem_alloc.h>

#include "constant.h"
namespace {
VulkanEngine *loadedEngine = nullptr;
auto constexpr bUseValidationLayers = true;
uint32_t get_minimum_surface_image_count(VkPhysicalDevice device,
                                         VkSurfaceKHR surface) {
  VkSurfaceCapabilitiesKHR cap;
  VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &cap));
  return cap.minImageCount;
}

VkComputePipelineCreateInfo
get_compute_pipeline_create_info(VkDevice _device, VkPipelineLayout layout,
                                 std::string_view shaderPath) {
  VkShaderModule computeDrawShader;

  auto absPath = (engine_constant::GetShaderRoot() / shaderPath);
  if (!vkutil::load_shader_module(_device, absPath.string(),
                                  &computeDrawShader)) {
    fmt::print(stderr,
               "Error when building the compute shader, Can't load "
               "{} \n",
               absPath.string());
  }

  VkPipelineShaderStageCreateInfo stageinfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .pNext = nullptr,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = computeDrawShader,
      .pName = "main"};

  VkComputePipelineCreateInfo pipelineCreateInfo{
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .pNext = nullptr,
      .stage = stageinfo,
      .layout = layout,
  };

  return pipelineCreateInfo;
}

}; // namespace

VulkanEngine &VulkanEngine::Get() { return *loadedEngine; }

void VulkanEngine::init() {
  // only one engine initialization is allowed with the application.
  assert(loadedEngine == nullptr);
  loadedEngine = this;

  // We initialize SDL and create a window with it.
  SDL_Init(SDL_INIT_VIDEO);

  SDL_WindowFlags window_flags =
      (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

  _window = SDL_CreateWindow("Vulkan Engine", SDL_WINDOWPOS_UNDEFINED,
                             SDL_WINDOWPOS_UNDEFINED, _windowExtent.width,
                             _windowExtent.height, window_flags);
  if (!_window) {
    fmt::println(stderr, "Can't Create window, SDLError: {}", SDL_GetError());
    std::abort();
  }

  init_vulkan();
  init_swapchain();
  init_commands();
  init_sync_structures();
  init_descriptors();
  init_pipelines();
  init_imgui();
  init_default_data();

  // everything went fine
  _isInitialized = true;
}
AllocatedBuffer VulkanEngine::create_buffer(size_t allocSize,
                                            VkBufferUsageFlags usage,
                                            VmaMemoryUsage memoryUsage) {
  VkBufferCreateInfo buffer_info{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .pNext = nullptr,
      .size = allocSize,
      .usage = usage,
  };

  VmaAllocationCreateInfo vmaalloc_info{
      .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT,
      .usage = memoryUsage,
  };
  AllocatedBuffer buffer;

  VK_CHECK(vmaCreateBuffer(_allocator, &buffer_info, &vmaalloc_info,
                           &buffer.buffer, &buffer.allocation, &buffer.info));
  return buffer;
};

void VulkanEngine::destroy_buffer(const AllocatedBuffer &buffer) {
  vmaDestroyBuffer(_allocator, buffer.buffer, buffer.allocation);
}

GPUMeshBuffer VulkanEngine::uploadMesh(std::span<uint32_t> indices,
                                       std::span<Vertex> vertices) {
  const size_t vertexBufferSize = vertices.size() * sizeof(Vertex);
  const size_t indexBufferSize = indices.size() * sizeof(uint32_t);

  GPUMeshBuffer newSurface;
  newSurface.vertexBuffer = create_buffer(
      vertexBufferSize,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      VMA_MEMORY_USAGE_GPU_ONLY);
  VkBufferDeviceAddressInfo deviceAdressInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
      .buffer = newSurface.vertexBuffer.buffer,
  };
  newSurface.VertexBufferAddress =
      vkGetBufferDeviceAddress(_device, &deviceAdressInfo);
  newSurface.indexBuffer = create_buffer(indexBufferSize,
                                         VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                         VMA_MEMORY_USAGE_GPU_ONLY);

  AllocatedBuffer staging = create_buffer(vertexBufferSize + indexBufferSize,
                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                          VMA_MEMORY_USAGE_CPU_ONLY);
  void *data = staging.allocation->GetMappedData();
  memcpy(data, vertices.data(), vertexBufferSize);
  memcpy(static_cast<uint8_t *>(data) + vertexBufferSize, indices.data(),
         indexBufferSize);

  immediate_submit([&](VkCommandBuffer cmd) {
    VkBufferCopy vertexCopy{
        .srcOffset = 0,
        .dstOffset = 0,
        .size = vertexBufferSize,
    };
    vkCmdCopyBuffer(cmd, staging.buffer, newSurface.vertexBuffer.buffer, 1,
                    &vertexCopy);
    VkBufferCopy indexCopy{
        .srcOffset = vertexBufferSize,
        .dstOffset = 0,
        .size = indexBufferSize,
    };
    vkCmdCopyBuffer(cmd, staging.buffer, newSurface.indexBuffer.buffer, 1,
                    &indexCopy);
  });
  destroy_buffer(staging);
  return newSurface;
};

AllocatedImage VulkanEngine::create_image(VkExtent3D size, VkFormat format,
                                          VkImageUsageFlags usage,
                                          bool mipmapped) {
  AllocatedImage newImage;
  newImage.imageFormat = format;
  newImage.imageExtent = size;

  VkImageCreateInfo img_info = vkinit::image_create_info(format, usage, size);
  if (mipmapped) {
    img_info.mipLevels = 1;
  }

  VmaAllocationCreateInfo alloc_info{};
  alloc_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;
  alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

  VK_CHECK(vmaCreateImage(_allocator, &img_info, &alloc_info, &newImage.image,
                          &newImage.allocation, nullptr));

  VkImageAspectFlags aspectFlag = (format == VK_FORMAT_D32_SFLOAT)
                                      ? VK_IMAGE_ASPECT_DEPTH_BIT
                                      : VK_IMAGE_ASPECT_COLOR_BIT;

  VkImageViewCreateInfo view_info =
      vkinit::imageview_create_info(format, newImage.image, aspectFlag);
  view_info.subresourceRange.levelCount = img_info.mipLevels;

  VK_CHECK(
      vkCreateImageView(_device, &view_info, nullptr, &newImage.imageView));
  return newImage;
}
AllocatedImage VulkanEngine::create_image(void *data, VkExtent3D size,
                                          VkFormat format,
                                          VkImageUsageFlags usage,
                                          bool mipmapped) {
  size_t data_size = size.depth * size.width * size.height * 4;
  AllocatedBuffer uploadbuffer = create_buffer(
      data_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

  memcpy(uploadbuffer.info.pMappedData, data, data_size);

  AllocatedImage new_image = create_image(
      size, format,
      usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      mipmapped);

  immediate_submit([&](VkCommandBuffer cmd) {
    vkutil::transition_image(cmd, new_image.image, VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    VkBufferImageCopy copyRegion = {};
    copyRegion.bufferOffset = 0;
    copyRegion.bufferRowLength = 0;
    copyRegion.bufferImageHeight = 0;

    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.mipLevel = 0;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageExtent = size;

    // copy the buffer into the image
    vkCmdCopyBufferToImage(cmd, uploadbuffer.buffer, new_image.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                           &copyRegion);

    vkutil::transition_image(cmd, new_image.image,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  });

  destroy_buffer(uploadbuffer);

  return new_image;
}

void VulkanEngine::destroy_image(const AllocatedImage &img) {
  vkDestroyImageView(_device, img.imageView, nullptr);
  vmaDestroyImage(_allocator, img.image, img.allocation);
}

void VulkanEngine::init_imgui() {
  VkDescriptorPoolSize pool_sizes[] = {
      {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
      {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}};

  VkDescriptorPoolCreateInfo pool_info{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
      .maxSets = 1000,
      .poolSizeCount = std::size(pool_sizes),
      .pPoolSizes = pool_sizes};

  VkDescriptorPool imguiPool;
  VK_CHECK(vkCreateDescriptorPool(_device, &pool_info, nullptr, &imguiPool));
  ImGui::CreateContext();
  ImGui_ImplSDL2_InitForVulkan(_window);
  ImGui_ImplVulkan_InitInfo init_info{
      .Instance = _instance,
      .PhysicalDevice = _chosenGPU,
      .Device = _device,
      .Queue = _graphicsQueue,
      .DescriptorPool = imguiPool,
      .MinImageCount = get_frame_overlap(),
      .ImageCount = get_frame_overlap(),
      .MSAASamples = VK_SAMPLE_COUNT_1_BIT,
      .UseDynamicRendering = true,
      .PipelineRenderingCreateInfo{
          .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
          .colorAttachmentCount = 1,
          .pColorAttachmentFormats = &_swapchainImageFormat,
      },
  };
  ImGui_ImplVulkan_Init(&init_info);
  ImGui_ImplVulkan_CreateFontsTexture();

  _mainDeletionQueue.push_function([=, this]() {
    ImGui_ImplVulkan_Shutdown();
    vkDestroyDescriptorPool(_device, imguiPool, nullptr);
  });
};

void VulkanEngine::draw_imgui(VkCommandBuffer cmd, VkImageView image) {
  VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(
      image, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  VkRenderingInfo renderInfo =
      vkinit::rendering_info(_swapchainExtent, &colorAttachment, nullptr);
  vkCmdBeginRendering(cmd, &renderInfo);

  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

  vkCmdEndRendering(cmd);
}

void VulkanEngine::init_pipelines() {
  init_background_pipeline();
  init_mesh_pipeline();
}

void VulkanEngine::init_default_data() {
  std::array<Vertex, 4> rect_vertices{};

  rect_vertices[0].position = glm::vec3(0.5f, -0.5f, 0.0f);
  rect_vertices[1].position = glm::vec3(0.5f, 0.5f, 0.0f);
  rect_vertices[2].position = glm::vec3(-0.5f, -0.5f, 0.0f);
  rect_vertices[3].position = glm::vec3(-0.5f, 0.5f, 0.0f);

  rect_vertices[0].color = {0, 0, 0, 1};
  rect_vertices[1].color = {0.5, 0.5, 0.5, 1};
  rect_vertices[2].color = {1, 0, 0, 1};
  rect_vertices[3].color = {0, 1, 0, 1};

  std::array<uint32_t, 6> rect_indices{};
  rect_indices[0] = 0;
  rect_indices[1] = 1;
  rect_indices[2] = 2;

  rect_indices[3] = 2;
  rect_indices[4] = 1;
  rect_indices[5] = 3;

  rectangle = uploadMesh(rect_indices, rect_vertices);
  auto absPath = (engine_constant::GetAssetRoot() / "basicmesh.glb");
  auto result = loadGltfMeshes(this, absPath);
  if (!result) {
    std::abort();
  }
  testMeshes = std::move(result.value());

  uint32_t white = glm::packUnorm4x8(glm::vec4(1, 1, 1, 1));
  _whiteImage =
      create_image((void *)&white, VkExtent3D{1, 1, 1},
                   VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

  uint32_t grey = glm::packUnorm4x8(glm::vec4(0.66f, 0.66f, 0.66f, 1));
  _greyImage =
      create_image((void *)&grey, VkExtent3D{1, 1, 1}, VK_FORMAT_R8G8B8A8_UNORM,
                   VK_IMAGE_USAGE_SAMPLED_BIT);

  uint32_t black = glm::packUnorm4x8(glm::vec4(0, 0, 0, 0));
  _blackImage =
      create_image((void *)&black, VkExtent3D{1, 1, 1},
                   VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

  uint32_t magenta = glm::packUnorm4x8(glm::vec4(0, 0, 0, 1));
  std::array<uint32_t, 16 * 16> pixels;

  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 16; j++) {
      pixels[i * 16 + j] = ((i % 2) ^ (j % 2)) ? magenta : black;
    }
  }

  _errorCheckImage =
      create_image(pixels.data(), VkExtent3D{16, 16, 1},
                   VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);
  VkSamplerCreateInfo samplerInfo{.sType =
                                      VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};

  samplerInfo.magFilter = VK_FILTER_NEAREST;
  samplerInfo.minFilter = VK_FILTER_NEAREST;

  vkCreateSampler(_device, &samplerInfo, nullptr, &_defaultSamplerNearest);

  samplerInfo.magFilter = VK_FILTER_LINEAR;
  samplerInfo.minFilter = VK_FILTER_LINEAR;

  vkCreateSampler(_device, &samplerInfo, nullptr, &_defaultSamplerLinear);

  _mainDeletionQueue.push_function([=, this]() {
    vkDestroySampler(_device, _defaultSamplerLinear, nullptr);
    vkDestroySampler(_device, _defaultSamplerNearest, nullptr);

    destroy_image(_whiteImage);
    destroy_image(_blackImage);
    destroy_image(_greyImage);
    destroy_image(_errorCheckImage);
  });

  _mainDeletionQueue.push_function([=, this]() {
    destroy_buffer(rectangle.indexBuffer);
    destroy_buffer(rectangle.vertexBuffer);
    for (auto &mesh : testMeshes) {
      destroy_buffer(mesh->meshBuffer.indexBuffer);
      destroy_buffer(mesh->meshBuffer.vertexBuffer);
    }
  });
}

void VulkanEngine::init_mesh_pipeline() {
  VkShaderModule vertexShader;
  VkShaderModule fragmentShader;
  auto fragAbsPath =
      (engine_constant::GetShaderRoot() / "tex_image_frag.spv");
  auto vertAbsPath =
      (engine_constant::GetShaderRoot() / "colored_triangle_mesh_vert.spv");
  if (!vkutil::load_shader_module(_device, fragAbsPath.string(),
                                  &fragmentShader)) {
    fmt::print(stderr,
               "Error when building the fragment shader, Can't load "
               "{} \n",
               fragAbsPath.string());
  }
  if (!vkutil::load_shader_module(_device, vertAbsPath.string(),
                                  &vertexShader)) {
    fmt::print(stderr,
               "Error when building the vertex shader, Can't load "
               "{} \n",
               vertAbsPath.string());
  }

  VkPushConstantRange bufferRange{
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
      .offset = 0,
      .size = sizeof(GPUDrawPushConstants),
  };

  VkPipelineLayoutCreateInfo pipeline_layout_info{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &_singleImageDescriptorLayout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &bufferRange,
  };
  VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr,
                                  &_meshPipelineLayout));

  vkutil::PipelineBuilder builder;
  builder._pipelineLayout = _meshPipelineLayout;
  builder.set_shader(vertexShader, fragmentShader);
  builder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  builder.set_polygon_mode(VK_POLYGON_MODE_FILL);
  builder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
  builder.set_multisample_none();
  builder.disable_blending();
  // builder.enable_blending_additive();
  builder.enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL);
  builder.set_color_attachment_format(_drawImage.imageFormat);
  builder.set_depth_format(_depthImage.imageFormat);
  _meshPipeline = builder.build_pipeline(_device);
  assert(_meshPipeline);

  vkDestroyShaderModule(_device, vertexShader, nullptr);
  vkDestroyShaderModule(_device, fragmentShader, nullptr);

  _mainDeletionQueue.push_function([=, this]() {
    vkDestroyPipelineLayout(_device, _meshPipelineLayout, nullptr);
    vkDestroyPipeline(_device, _meshPipeline, nullptr);
  });
}

void VulkanEngine::init_background_pipeline() {

  VkPushConstantRange pushConstantRange{
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0,
      .size = sizeof(ComputePushConstants),
  };
  VkPipelineLayoutCreateInfo computeLayout{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pNext = nullptr,
      .setLayoutCount = 1,
      .pSetLayouts = &_drawImageDescriptorLayout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pushConstantRange,
  };

  VK_CHECK(vkCreatePipelineLayout(_device, &computeLayout, nullptr,
                                  &_gradientPipelineLayout));
  auto gradientPipelineCreateInfo = get_compute_pipeline_create_info(
      _device, _gradientPipelineLayout, "gradient_color.spv");

  auto skyPipelineCreateInfo = get_compute_pipeline_create_info(
      _device, _gradientPipelineLayout, "sky.spv");

  ComputeEffect gradient{
      .name = "gradient",
      .layout = _gradientPipelineLayout,
      .data =
          {
              .data1 = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),
              .data2 = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f),
          },
  };

  ComputeEffect sky{
      .name = "sky",
      .layout = _gradientPipelineLayout,
      .data =
          {
              .data1 = glm::vec4(0.1f, 0.2f, 0.4f, 0.97f),
          },
  };

  VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1,
                                    &gradientPipelineCreateInfo, nullptr,
                                    &gradient.pipeline));

  VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1,
                                    &skyPipelineCreateInfo, nullptr,
                                    &sky.pipeline));
  backgroundEffects.push_back(gradient);
  backgroundEffects.push_back(sky);

  vkDestroyShaderModule(_device, gradientPipelineCreateInfo.stage.module,
                        nullptr);
  vkDestroyShaderModule(_device, skyPipelineCreateInfo.stage.module, nullptr);
  _mainDeletionQueue.push_function([=, this]() {
    vkDestroyPipelineLayout(_device, _gradientPipelineLayout, nullptr);
    vkDestroyPipeline(_device, sky.pipeline, nullptr);
    vkDestroyPipeline(_device, gradient.pipeline, nullptr);
  });
}

void VulkanEngine::init_descriptors() {
  std::vector<DescriptorAllocator::PoolSizeRatio> sizes{
      {.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .ratio = 1}};

  _globalDescriptorAllocator.init_pool(_device, 10, sizes);

  {
    DescriptorLayoutBuilder builder;
    builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    _drawImageDescriptorLayout =
        builder.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);
  }
  _drawImageDescriptors =
      _globalDescriptorAllocator.allocate(_device, _drawImageDescriptorLayout);

  VkDescriptorImageInfo imgInfo{
      .imageView = _drawImage.imageView,
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
  };

  VkWriteDescriptorSet drawImageWrite{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .pNext = nullptr,
      .dstSet = _drawImageDescriptors,
      .dstBinding = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .pImageInfo = &imgInfo,
  };
  vkUpdateDescriptorSets(_device, 1, &drawImageWrite, 0, nullptr);

  for (int i = 0; i < get_frame_overlap(); i++) {
    // create a descriptor pool
    std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> frame_sizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4},
    };

    _frames[i]._frameDescriptor = DescriptorAllocatorGrowable{};
    _frames[i]._frameDescriptor.init(_device, 1000, frame_sizes);

    _mainDeletionQueue.push_function(
        [&, i]() { _frames[i]._frameDescriptor.destroy_pools(_device); });
  }
  {
    DescriptorLayoutBuilder builder;
    builder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    _gpuSenceDataDescriptorLayout = builder.build(
        _device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
  }
  {
    DescriptorLayoutBuilder builder;
    builder.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    _singleImageDescriptorLayout =
        builder.build(_device, VK_SHADER_STAGE_FRAGMENT_BIT);
  }

  _mainDeletionQueue.push_function([=, this]() {
    _globalDescriptorAllocator.destroy_pool(_device);
    vkDestroyDescriptorSetLayout(_device, _drawImageDescriptorLayout, nullptr);
    vkDestroyDescriptorSetLayout(_device, _gpuSenceDataDescriptorLayout,
                                 nullptr);
    vkDestroyDescriptorSetLayout(_device, _singleImageDescriptorLayout,
                                 nullptr);
  });
}

void VulkanEngine::init_vulkan() {
  vkb::InstanceBuilder builder;
  auto const inst_ret = builder.set_app_name("Example Vulkan Application")
                            .request_validation_layers(bUseValidationLayers)
                            .use_default_debug_messenger()
                            .require_api_version(1, 3, 0)
                            .build();

  RESULT_CHECK(inst_ret, "Can't Init Vulkan Instance, Error: {}");
  _instance = inst_ret->instance;
  _debug_messager = inst_ret->debug_messenger;

  if (SDL_Vulkan_CreateSurface(_window, _instance, &_surface) != SDL_TRUE) {
    fmt::println(stderr, "Can't create Vulkan Surface, error: {}",
                 SDL_GetError());
    std::abort();
  }

  VkPhysicalDeviceVulkan13Features features_13{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .synchronization2 = true,
      .dynamicRendering = true,
  };
  VkPhysicalDeviceVulkan12Features features_12{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .descriptorIndexing = true,
      .bufferDeviceAddress = true,
  };

  vkb::PhysicalDeviceSelector selector{*inst_ret, _surface};

  vkb::Result<vkb::PhysicalDevice> physicalDevice =
      selector.set_minimum_version(1, 3)
          .set_required_features_13(features_13)
          .set_required_features_12(features_12)
          .select();

  RESULT_CHECK(physicalDevice,
               "Can't select suitable PhysicalDevice, errormessage: {}");

  fmt::println("Currently Device maxPushConstantsSize: {}",
               physicalDevice->properties.limits.maxPushConstantsSize);

  vkb::DeviceBuilder deviceBuilder{*physicalDevice};
  vkb::Result<vkb::Device> vkbDevice = deviceBuilder.build();

  RESULT_CHECK(vkbDevice, "Failed to create physical device, errormessage: {}");

  _device = vkbDevice->device;
  _chosenGPU = vkbDevice->physical_device;

  auto graphicsQueueResult = vkbDevice->get_queue(vkb::QueueType::graphics);
  RESULT_CHECK(graphicsQueueResult,
               "Fail to get graphics queue, errormessage: {}");

  _graphicsQueue = *graphicsQueueResult;

  auto graphicsQueueFamilyResult =
      vkbDevice->get_queue_index(vkb::QueueType::graphics);
  RESULT_CHECK(graphicsQueueFamilyResult,
               "Failed to get Graphics Queue Family, errormessage: {}");

  _graphicsQueueFamily = *graphicsQueueFamilyResult;
  VmaAllocatorCreateInfo allocatorInfo{
      .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
      .physicalDevice = _chosenGPU,
      .device = _device,
      .instance = _instance,
  };
  vmaCreateAllocator(&allocatorInfo, &_allocator);
  _mainDeletionQueue.push_function(
      [this]() { vmaDestroyAllocator(_allocator); });
}

void VulkanEngine::create_swapchain(uint32_t width, uint32_t height) {
  vkb::SwapchainBuilder swapchainBuilder{_chosenGPU, _device, _surface};
  _swapchainImageFormat = VK_FORMAT_B8G8R8A8_UNORM;
  uint32_t minImageCount =
      get_minimum_surface_image_count(_chosenGPU, _surface) + 1;
  vkb::Result<vkb::Swapchain> vkbSwapchain =
      swapchainBuilder
          .set_desired_format(VkSurfaceFormatKHR{
              .format = _swapchainImageFormat,
              .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
          .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
          .set_desired_extent(width, height)
          .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
          .set_desired_min_image_count(minImageCount)
          .build();

  RESULT_CHECK(vkbSwapchain, "Failed to create swapchain, errormessage: {}");
  _swapchainExtent = vkbSwapchain->extent;
  _swapchain = vkbSwapchain->swapchain;

  auto images = vkbSwapchain->get_images();
  RESULT_CHECK(images, "Failed to get image, errormessage: {}");
  _swapchainImage = *images;

  auto image_views = vkbSwapchain->get_image_views();

  RESULT_CHECK(image_views, "Failed to get image view, errormessage: {}");

  _swapchainImageViews = *image_views;
  _frames.resize(_swapchainImage.size());
}

void VulkanEngine::resize_swapchain() {
  vkDeviceWaitIdle(_device);
  destroy_swapchain();

  fmt::println("Recreating Swapchain");
  int32_t w, h;
  SDL_GetWindowSize(_window, &w, &h);
  _windowExtent.width = w;
  _windowExtent.height = h;
  create_swapchain(_windowExtent.width, _windowExtent.height);

  resize_requested = false;
}

void VulkanEngine::init_swapchain() {
  create_swapchain(_windowExtent.width, _windowExtent.height);
  VkExtent3D drawImageExtent{
      .width = _windowExtent.width,
      .height = _windowExtent.height,
      .depth = 1,
  };
  _drawImage.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
  _drawImage.imageExtent = drawImageExtent;

  VkImageUsageFlags drawImageUsages{};
  drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  drawImageUsages |= VK_IMAGE_USAGE_STORAGE_BIT;
  drawImageUsages |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

  VkImageCreateInfo rimg_info = vkinit::image_create_info(
      _drawImage.imageFormat, drawImageUsages, drawImageExtent);
  // GPU VRAM Contain an "Upload Heap" it can be accessed by CPU with PCIE Bus,
  // And that heap size can be control in BIOS or driver panel call "resizable
  // bar".

  // clang-format off
  // So Memory layout like this:
  //                  CPU Ram                                     |               GPU VRAM
  //  Host Only Memory  | Device can access Host Memory | - Connect by PCIE -  | Upload Heap | rest of GPU VRAM
  // clang-format on
  VmaAllocationCreateInfo rimg_allocinfo{
      .usage = VMA_MEMORY_USAGE_GPU_ONLY,
      .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
  vmaCreateImage(_allocator, &rimg_info, &rimg_allocinfo, &_drawImage.image,
                 &_drawImage.allocation, nullptr);
  VkImageViewCreateInfo rview_info = vkinit::imageview_create_info(
      _drawImage.imageFormat, _drawImage.image, VK_IMAGE_ASPECT_COLOR_BIT);

  VK_CHECK(
      vkCreateImageView(_device, &rview_info, nullptr, &_drawImage.imageView));

  _depthImage.imageFormat = VK_FORMAT_D32_SFLOAT;
  _depthImage.imageExtent = drawImageExtent;
  VkImageUsageFlags depthImageUsages{};
  depthImageUsages |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

  VkImageCreateInfo dimg_info = vkinit::image_create_info(
      _depthImage.imageFormat, depthImageUsages, drawImageExtent);

  vmaCreateImage(_allocator, &dimg_info, &rimg_allocinfo, &_depthImage.image,
                 &_depthImage.allocation, nullptr);
  VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(
      _depthImage.imageFormat, _depthImage.image, VK_IMAGE_ASPECT_DEPTH_BIT);

  VK_CHECK(
      vkCreateImageView(_device, &dview_info, nullptr, &_depthImage.imageView));

  _mainDeletionQueue.push_function([=, this]() {
    vkDestroyImageView(_device, _drawImage.imageView, nullptr);
    vmaDestroyImage(_allocator, _drawImage.image, _drawImage.allocation);
    vkDestroyImageView(_device, _depthImage.imageView, nullptr);
    vmaDestroyImage(_allocator, _depthImage.image, _depthImage.allocation);
  });
}

void VulkanEngine::destroy_swapchain() {
  vkDestroySwapchainKHR(_device, _swapchain, nullptr);

  for (auto &imgView : _swapchainImageViews) {
    vkDestroyImageView(_device, imgView, nullptr);
  }
}

void VulkanEngine::init_commands() {
  VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(
      _graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

  for (int i = 0; i < get_frame_overlap(); i++) {
    VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr,
                                 &_frames[i]._commandPool));

    VkCommandBufferAllocateInfo cmdAllocInfo =
        vkinit::command_buffer_allocate_info(_frames[i]._commandPool, 1);

    VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo,
                                      &_frames[i]._mainCommandBuffer));
  }

  VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr,
                               &_immCommandPool));
  VkCommandBufferAllocateInfo cmdAllocInfo =
      vkinit::command_buffer_allocate_info(_immCommandPool, 1);
  VK_CHECK(
      vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_immCommandBuffer));

  _mainDeletionQueue.push_function(
      [=, this]() { vkDestroyCommandPool(_device, _immCommandPool, nullptr); });
}

void VulkanEngine::init_sync_structures() {
  VkFenceCreateInfo fenceInfo =
      vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
  VkSemaphoreCreateInfo semaphoreInfo = vkinit::semaphore_create_info();
  for (int i = 0; i < get_frame_overlap(); i++) {
    VK_CHECK(
        vkCreateFence(_device, &fenceInfo, nullptr, &_frames[i]._renderFence));
    VK_CHECK(vkCreateSemaphore(_device, &semaphoreInfo, nullptr,
                               &_frames[i]._renderSemaphore));
    VK_CHECK(vkCreateSemaphore(_device, &semaphoreInfo, nullptr,
                               &_frames[i]._swapchainSemaphore));
  }
  VK_CHECK(vkCreateFence(_device, &fenceInfo, nullptr, &_immFence));
  _mainDeletionQueue.push_function(
      [=, this]() { vkDestroyFence(_device, _immFence, nullptr); });
}

void VulkanEngine::immediate_submit(
    std::function<void(VkCommandBuffer cmd)> &&function) {
  VK_CHECK(vkResetFences(_device, 1, &_immFence));
  VK_CHECK(vkResetCommandBuffer(_immCommandBuffer, 0));

  VkCommandBuffer cmd = _immCommandBuffer;
  VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(
      VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
  VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));
  function(cmd);
  VK_CHECK(vkEndCommandBuffer(cmd));
  VkCommandBufferSubmitInfo cmdSubmitInfo =
      vkinit::command_buffer_submit_info(cmd);
  VkSubmitInfo2 submitInfo =
      vkinit::submit_info(&cmdSubmitInfo, nullptr, nullptr);

  VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submitInfo, _immFence));
  VK_CHECK(vkWaitForFences(_device, 1, &_immFence, true, 9999999999));
};

void VulkanEngine::cleanup() {
  if (_isInitialized) {
    vkDeviceWaitIdle(_device);
    for (int i = 0; i < get_frame_overlap(); i++) {
      vkDestroyCommandPool(_device, _frames[i]._commandPool, nullptr);
      vkDestroyFence(_device, _frames[i]._renderFence, nullptr);
      vkDestroySemaphore(_device, _frames[i]._renderSemaphore, nullptr);
      vkDestroySemaphore(_device, _frames[i]._swapchainSemaphore, nullptr);
      _frames[i]._deletionQueue.flush();
    }
    _mainDeletionQueue.flush();

    destroy_swapchain();
    vkDestroySurfaceKHR(_instance, _surface, nullptr);
    vkDestroyDevice(_device, nullptr);

    vkb::destroy_debug_utils_messenger(_instance, _debug_messager);
    vkDestroyInstance(_instance, nullptr);
    SDL_DestroyWindow(_window);
  }

  // clear engine pointer
  loadedEngine = nullptr;
}
void VulkanEngine::draw_background(VkCommandBuffer cmd) {
  ComputeEffect &effect = backgroundEffects[currentEffectIndex];
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, effect.pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          _gradientPipelineLayout, 0, 1, &_drawImageDescriptors,
                          0, nullptr);
  vkCmdPushConstants(cmd, _gradientPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                     0, sizeof(ComputePushConstants), &effect.data);
  vkCmdDispatch(cmd, std::ceil(float(_drawImage.imageExtent.width) / 16.0f),
                std::ceil(float(_drawImage.imageExtent.height) / 16.0f), 1);
}

void VulkanEngine::draw_geometry(VkCommandBuffer cmd) {
  // allocate a new uniform buffer for the scene data
  AllocatedBuffer gpuSceneDataBuffer =
      create_buffer(sizeof(GPUSceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    VMA_MEMORY_USAGE_CPU_TO_GPU);

  // add it to the deletion queue of this frame so it gets deleted once its been
  // used
  get_current_frame()._deletionQueue.push_function(
      [=, this]() { destroy_buffer(gpuSceneDataBuffer); });

  // write the buffer
  GPUSceneData *sceneUniformData =
      (GPUSceneData *)gpuSceneDataBuffer.allocation->GetMappedData();
  *sceneUniformData = sceneData;

  // create a descriptor set that binds that buffer and update it
  VkDescriptorSet globalDescriptor =
      get_current_frame()._frameDescriptor.allocate(
          _device, _gpuSenceDataDescriptorLayout);

  DescriptorWriter writer;
  writer.write_buffer(0, gpuSceneDataBuffer.buffer, sizeof(GPUSceneData), 0,
                      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
  writer.update_set(_device, globalDescriptor);

  VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(
      _drawImage.imageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  VkRenderingAttachmentInfo depthAttachment = vkinit::depth_attachment_info(
      _depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

  VkRenderingInfo renderInfo =
      vkinit::rendering_info(_drawExtent, &colorAttachment, &depthAttachment);
  vkCmdBeginRendering(cmd, &renderInfo);


  VkViewport viewport{
      .x = 0.0f,
      .y = 0.0f,
      .width = static_cast<float>(_drawExtent.width),
      .height = static_cast<float>(_drawExtent.height),
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
  };

  vkCmdSetViewport(cmd, 0, 1, &viewport);
  VkRect2D scissor{
      .offset = {0, 0},
      .extent = _drawExtent,
  };
  vkCmdSetScissor(cmd, 0, 1, &scissor);

  glm::mat4 view =
      glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -5.0f));
  glm::mat4 projection = glm::perspective(
      glm::radians(70.f), (float)_drawExtent.width / (float)_drawExtent.height,
      10000.f, 0.1f);

  projection[1][1] *= -1;

  GPUDrawPushConstants push_constants{
      .worldMatrix = projection * view,
      .vertexBuffer = testMeshes[2]->meshBuffer.VertexBufferAddress,
  };

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _meshPipeline);
  VkDescriptorSet imageSet = get_current_frame()._frameDescriptor.allocate(_device, _singleImageDescriptorLayout);
  {
      DescriptorWriter writer;
      writer.write_image(0, _whiteImage.imageView, _defaultSamplerNearest, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

      writer.update_set(_device, imageSet);
  }

  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _meshPipelineLayout, 0, 1, &imageSet, 0, nullptr);

  vkCmdPushConstants(cmd, _meshPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                     sizeof(GPUDrawPushConstants), &push_constants);
  vkCmdBindIndexBuffer(cmd, testMeshes[2]->meshBuffer.indexBuffer.buffer, 0,
                       VK_INDEX_TYPE_UINT32);
  vkCmdDrawIndexed(cmd, testMeshes[2]->surfaces[0].count, 1,
                   testMeshes[2]->surfaces[0].startIndex, 0, 0);

  vkCmdEndRendering(cmd);
}

void VulkanEngine::draw() {
  get_current_frame()._deletionQueue.flush();
  get_current_frame()._frameDescriptor.clear_pools(_device);

  VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._renderFence, true,
                           SecondsInNano(100)));
  VK_CHECK(vkResetFences(_device, 1, &get_current_frame()._renderFence));

  uint32_t swapchainImageIndex;
  {
    auto e = vkAcquireNextImageKHR(_device, _swapchain, UINT64_MAX,
                                   get_current_frame()._swapchainSemaphore,
                                   VK_NULL_HANDLE, &swapchainImageIndex);
    if (e == VK_ERROR_OUT_OF_DATE_KHR || e == VK_SUBOPTIMAL_KHR) {
      // need to handle _renderFence are been signaled, so we need to reset it.
      // and need to waite _swapchainSemaphore complete.
      VkCommandBuffer cmd = get_current_frame()._mainCommandBuffer;
      VK_CHECK(vkResetCommandBuffer(cmd, 0));

      VkCommandBufferBeginInfo beginInfo = vkinit::command_buffer_begin_info(
          VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
      VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

      VkCommandBufferSubmitInfo cmdinfo =
          vkinit::command_buffer_submit_info(cmd);
      VK_CHECK(vkEndCommandBuffer(cmd));
      VkSemaphoreSubmitInfo waitInfo = vkinit::semaphore_submit_info(
          VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR,
          get_current_frame()._swapchainSemaphore);

      VkSubmitInfo2 submit = vkinit::submit_info(&cmdinfo, nullptr, &waitInfo);
      VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit,
                              get_current_frame()._renderFence));
      resize_requested = true;
      return;
    }
  }

  // TODO: Rewrite the sync way.
  //  WIndows Nvidia FIFO will cause vkAcquireNextImageKHR impl not block
  //  until next available presentImage, vkAcquireNextImageKHR never
  //  block image, it block on QueuePresent.
  //  so it will return same imageIndex in two frame.
  //  and current our synchronize way need to assert that swapchainImageIndex
  //  always +1 % totalImageSize. so we change windows presentMode to MAILBOX.

  get_current_frame()._deletionQueue.flush();
  get_current_frame()._frameDescriptor.clear_pools(_device);

  VkCommandBuffer cmd = get_current_frame()._mainCommandBuffer;
  VK_CHECK(vkResetCommandBuffer(cmd, 0));

  VkCommandBufferBeginInfo beginInfo = vkinit::command_buffer_begin_info(
      VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

  _drawExtent.width =
      std::min(_swapchainExtent.width, _drawImage.imageExtent.width) *
      rendrScale;
  _drawExtent.height =
      std::min(_swapchainExtent.height, _drawImage.imageExtent.height) *
      rendrScale;
  VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

  vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_UNDEFINED,
                           VK_IMAGE_LAYOUT_GENERAL);

  // draw command
  draw_background(cmd);

  vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_GENERAL,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

  vkutil::transition_image(cmd, _depthImage.image, VK_IMAGE_LAYOUT_UNDEFINED,
                           VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
  draw_geometry(cmd);

  vkutil::transition_image(cmd, _drawImage.image,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
  vkutil::transition_image(cmd, _swapchainImage[swapchainImageIndex],
                           VK_IMAGE_LAYOUT_UNDEFINED,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  vkutil::copy_image_to_image(cmd, _drawImage.image,
                              _swapchainImage[swapchainImageIndex], _drawExtent,
                              _swapchainExtent);
  vkutil::transition_image(cmd, _swapchainImage[swapchainImageIndex],
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  draw_imgui(cmd, _swapchainImageViews[swapchainImageIndex]);
  vkutil::transition_image(cmd, _swapchainImage[swapchainImageIndex],
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                           VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

  VK_CHECK(vkEndCommandBuffer(cmd));
  VkCommandBufferSubmitInfo cmdinfo = vkinit::command_buffer_submit_info(cmd);
  VkSemaphoreSubmitInfo waitInfo = vkinit::semaphore_submit_info(
      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR,
      get_current_frame()._swapchainSemaphore);
  VkSemaphoreSubmitInfo signalInfo =
      vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
                                    get_current_frame()._renderSemaphore);
  VkSubmitInfo2 submit = vkinit::submit_info(&cmdinfo, &signalInfo, &waitInfo);
  VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit,
                          get_current_frame()._renderFence));

  VkPresentInfoKHR presentInfo = {
      .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
      .pNext = nullptr,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &get_current_frame()._renderSemaphore,
      .swapchainCount = 1,
      .pSwapchains = &_swapchain,
      .pImageIndices = &swapchainImageIndex,
  };
  {
    auto e = vkQueuePresentKHR(_graphicsQueue, &presentInfo);
    if (e == VK_ERROR_OUT_OF_DATE_KHR || e == VK_SUBOPTIMAL_KHR) {
      resize_requested = true;
    }
  }

  _frameNumber++;
}

void VulkanEngine::run() {
  SDL_Event e;
  bool bQuit = false;

  // main loop
  while (!bQuit) {
    // Handle events on queue
    while (SDL_PollEvent(&e) != 0) {
      // close the window when user alt-f4s or clicks the X button
      if (e.type == SDL_QUIT)
        bQuit = true;

      if (e.type == SDL_WINDOWEVENT) {
        if (e.window.event == SDL_WINDOWEVENT_MINIMIZED) {
          stop_rendering = true;
        }
        if (e.window.event == SDL_WINDOWEVENT_RESTORED) {
          stop_rendering = false;
        }
      }

      ImGui_ImplSDL2_ProcessEvent(&e);
    }

    // do not draw if we are minimized
    if (stop_rendering) {
      // throttle the speed to avoid the endless spinning
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }
    if (resize_requested) {
      resize_swapchain();
    }
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();
    if (ImGui::Begin("background")) {
      ImGui::SliderFloat("Render Scale", &rendrScale, 0.3f, 1.0f);
      ComputeEffect &effect = backgroundEffects[currentEffectIndex];
      ImGui::Text("Selected effect: %s", effect.name);
      ImGui::SliderInt("Effect Index", &currentEffectIndex, 0,
                       backgroundEffects.size() - 1);

      ImGui::InputFloat4("data1", (float *)&effect.data.data1);
      ImGui::InputFloat4("data2", (float *)&effect.data.data2);
      ImGui::InputFloat4("data3", (float *)&effect.data.data3);
      ImGui::InputFloat4("data4", (float *)&effect.data.data4);
    }
    ImGui::End();
    ImGui::Render();

    draw();
  }
}
