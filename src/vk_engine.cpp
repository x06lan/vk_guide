//> includes
#include "vk_engine.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <vk_images.h>
#include <vk_initializers.h>
#include <vk_pipelines.h>
#include <vk_types.h>

#include <chrono>
#include <thread>
#include <vulkan/vulkan_core.h>

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

#define VMA_IMPLEMENTATION
#include "VkBootstrap.h"
#include "vk_mem_alloc.h"

// static variables and constants
namespace {
    VulkanEngine *loadedEngine = nullptr;
    auto constexpr bUseValidationLayers = true;

    inline uint32_t get_surface_min_image_count(VkPhysicalDevice _chosenGPU,
                                                VkSurfaceKHR _surface) {
      VkSurfaceCapabilitiesKHR surfaceCapabilities;
      VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(_chosenGPU, _surface,
                                                         &surfaceCapabilities));
      return surfaceCapabilities.minImageCount;
    }

    VkComputePipelineCreateInfo computePipelinesInfo( VkDevice device,VkPipelineLayout gradientPipelineLayout,const char* shaderPath){

      VkShaderModule shader;
      if (!vkutil::load_shader_module(shaderPath, device, &shader)) {
          fmt::print("Error when building the compute shader \n");
      }
      VkPipelineShaderStageCreateInfo stageinfo{};
      stageinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      stageinfo.pNext = nullptr;
      stageinfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      stageinfo.module = shader;
      stageinfo.pName = "main";

      VkComputePipelineCreateInfo computePipelineCreateInfo{};
      computePipelineCreateInfo.sType =
          VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
      computePipelineCreateInfo.pNext = nullptr;
      computePipelineCreateInfo.layout = gradientPipelineLayout;
      computePipelineCreateInfo.stage = stageinfo;

      return computePipelineCreateInfo;
    };
}; // namespace

VulkanEngine &VulkanEngine::Get() { return *loadedEngine; }

void VulkanEngine::init() {
  // only one engine initialization is allowed with the application.
  assert(loadedEngine == nullptr);
  loadedEngine = this;

  // We initialize SDL and create a window with it.
  SDL_Init(SDL_INIT_VIDEO);

  SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);

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

  // everything went fine
  _isInitialized = true;
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
  fmt::println("MaxPushConstantsSize: {}",physicalDevice->properties.limits.maxPushConstantsSize );

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

  VmaAllocatorCreateInfo allocatorInfo = {};
  allocatorInfo.physicalDevice = _chosenGPU;
  allocatorInfo.device = _device;
  allocatorInfo.instance = _instance;
  allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
  vmaCreateAllocator(&allocatorInfo, &_allocator);

  _mainDeletionQueue.push_function([&]() { vmaDestroyAllocator(_allocator); });
}

void VulkanEngine::init_pipelines() { 
    init_background_pipelines(); 
}

void VulkanEngine::init_background_pipelines() {
  VkPipelineLayoutCreateInfo computeLayout{};
  computeLayout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  computeLayout.pNext = nullptr;
  computeLayout.pSetLayouts = &_drawImageDescriptorLayout;
  computeLayout.setLayoutCount = 1;

  VkPushConstantRange pushConstant{};
  pushConstant.offset = 0;
  pushConstant.size = sizeof(ComputePushConstants) ;
  pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  computeLayout.pPushConstantRanges = &pushConstant;
  computeLayout.pushConstantRangeCount = 1;

  VK_CHECK(vkCreatePipelineLayout(_device, &computeLayout, nullptr,
                                  &_gradientPipelineLayout));

  auto gradientPipelineInfo= computePipelinesInfo(_device,_gradientPipelineLayout,"./shaders/gradient_color.spv");
  auto skyPipelineInfo = computePipelinesInfo(_device,_gradientPipelineLayout,"./shaders/sky.spv");

  ComputeEffect gradientEffect{
      .name = "gradient",
      .layout = _gradientPipelineLayout,
      .data = {
          .data1 = glm::vec4(1, 0, 0, 1),
          .data2 = glm::vec4(0, 0, 1, 1),
      }
  };
  ComputeEffect skyEffect{
      .name = "sky",
      .layout = _gradientPipelineLayout,
  };


  VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1,
                                    &gradientPipelineInfo, nullptr,
                                    &gradientEffect.pipeline));

  VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1,
                                    &skyPipelineInfo, nullptr,
                                    &skyEffect.pipeline));

  backgroundEffects.push_back(gradientEffect);
  backgroundEffects.push_back(skyEffect);

  vkDestroyShaderModule(_device,  gradientPipelineInfo.stage.module, nullptr);
  vkDestroyShaderModule(_device,  skyPipelineInfo.stage.module, nullptr);


  _mainDeletionQueue.push_function([&]() {
    vkDestroyPipelineLayout(_device, _gradientPipelineLayout, nullptr);
    vkDestroyPipeline(_device, gradientEffect.pipeline, nullptr);
    vkDestroyPipeline(_device, skyEffect.pipeline, nullptr);
  });
}

void VulkanEngine::create_swapchain(uint32_t width, uint32_t height) {
  vkb::SwapchainBuilder swapchainBuilder{_chosenGPU, _device, _surface};
  _swapchainImageFormat = VK_FORMAT_B8G8R8A8_UNORM;

  // Get the frame overlap from the physical device and surface capabilities
  uint32_t frameOverlap = get_surface_min_image_count(_chosenGPU, _surface) + 1;
  fmt::println("Frame overlap: {}", frameOverlap);
  _frames.resize(static_cast<size_t>(frameOverlap));

  vkb::Result<vkb::Swapchain> vkbSwapchain =
      swapchainBuilder
          .set_desired_format(VkSurfaceFormatKHR{
              .format = _swapchainImageFormat,
              .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
          .set_desired_min_image_count(frameOverlap)
          .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
          .set_desired_extent(width, height)
          .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
          .build();

  RESULT_CHECK(vkbSwapchain, "Failed to create swapchain, errormessage: {}");

  _swapchainExtent = vkbSwapchain->extent;
  _swapchain = vkbSwapchain->swapchain;
  fmt::println("Swapchain created with extent: {}x{}", _swapchainExtent.width,
               _swapchainExtent.height);

  auto images = vkbSwapchain->get_images();
  RESULT_CHECK(images, "Failed to get image, errormessage: {}");
  _swapchainImage = *images;

  auto image_views = vkbSwapchain->get_image_views();
  RESULT_CHECK(image_views, "Failed to get image view, errormessage: {}");

  _swapchainImageViews = *image_views;
}

void VulkanEngine::init_swapchain() {
  create_swapchain(_windowExtent.width, _windowExtent.height);
  // draw image size will match the window
  VkExtent3D drawImageExtent = {_windowExtent.width, _windowExtent.height, 1};

  // hardcoding the draw format to 32 bit float
  _drawImage.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
  _drawImage.imageExtent = drawImageExtent;

  VkImageUsageFlags drawImageUsages{};
  drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  drawImageUsages |= VK_IMAGE_USAGE_STORAGE_BIT;
  drawImageUsages |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

  VkImageCreateInfo rimg_info = vkinit::image_create_info(
      _drawImage.imageFormat, drawImageUsages, drawImageExtent);

  // for the draw image, we want to allocate it from gpu local memory
  VmaAllocationCreateInfo rimg_allocinfo = {};
  rimg_allocinfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
  rimg_allocinfo.requiredFlags =
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT; // this mean the memory will be
                                           // allocated on the side that CPU can
                                           // not access

  // allocate and create the image
  vmaCreateImage(_allocator, &rimg_info, &rimg_allocinfo, &_drawImage.image,
                 &_drawImage.allocation, nullptr);

  // build a image-view for the draw image to use for rendering
  VkImageViewCreateInfo rview_info = vkinit::imageview_create_info(
      _drawImage.imageFormat, _drawImage.image, VK_IMAGE_ASPECT_COLOR_BIT);

  VK_CHECK(
      vkCreateImageView(_device, &rview_info, nullptr, &_drawImage.imageView));

  // add to deletion queues
  _mainDeletionQueue.push_function([=, this]() {
    vkDestroyImageView(_device, _drawImage.imageView, nullptr);
    vmaDestroyImage(_allocator, _drawImage.image, _drawImage.allocation);
  });
}

void VulkanEngine::destory_swapchain() {
  vkDestroySwapchainKHR(_device, _swapchain, nullptr);

  for (auto &imgView : _swapchainImageViews) {
    vkDestroyImageView(_device, imgView, nullptr);
  }
}

void VulkanEngine::init_commands() {
  VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(
      _graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

  for (size_t i = 0; i < get_frame_overlay(); i++) // Start from i = 0
  {
    VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr,
                                 &_frames[i]._commandPool));

    VkCommandBufferAllocateInfo cmdAllocInfo =
        vkinit::command_buffer_allocate_info(_frames[i]._commandPool, 1);

    VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo,
                                      &_frames[i]._mainCommandBuffer));
  }
  VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr,
                               &_immCommandPool));

  // allocate the command buffer for immediate submits
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
  for (int i = 0; i < get_frame_overlay(); i++) {
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
void VulkanEngine::init_descriptors() {
  std::vector<DescriptorAllocator::PoolSizeRatio> sizes = {
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
  };

  globalDescriptorAllocator.init_pool(_device, 10, sizes);

  {
    DescriptorLayoutBuilder builder;
    builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    _drawImageDescriptorLayout =
        builder.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);
  }

  _drawImageDescriptors =
      globalDescriptorAllocator.allocate(_device, _drawImageDescriptorLayout);

  VkDescriptorImageInfo imgInfo{};
  imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  imgInfo.imageView = _drawImage.imageView;

  VkWriteDescriptorSet drawImageWrite = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .pNext = nullptr,
      .dstSet = _drawImageDescriptors,
      .dstBinding = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .pImageInfo = &imgInfo,
  };
  vkUpdateDescriptorSets(_device, 1, &drawImageWrite, 0, nullptr);

  // make sure both the descriptor allocator and the new layout get cleaned up
  // properly
  _mainDeletionQueue.push_function([&]() {
    globalDescriptorAllocator.destroy_pool(_device);

    vkDestroyDescriptorSetLayout(_device, _drawImageDescriptorLayout, nullptr);
  });
}

void VulkanEngine::init_imgui() {
  // 1: create descriptor pool for IMGUI
  //  the size of the pool is very oversize, but it's copied from imgui demo
  //  itself.
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

  VkDescriptorPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  pool_info.maxSets = 1000;
  pool_info.poolSizeCount = (uint32_t)std::size(pool_sizes);
  pool_info.pPoolSizes = pool_sizes;

  VkDescriptorPool imguiPool;
  VK_CHECK(vkCreateDescriptorPool(_device, &pool_info, nullptr, &imguiPool));

  // 2: initialize imgui library

  // this initializes the core structures of imgui
  ImGui::CreateContext();

  // this initializes imgui for SDL
  ImGui_ImplSDL2_InitForVulkan(_window);

  // this initializes imgui for Vulkan
  ImGui_ImplVulkan_InitInfo init_info = {};
  init_info.Instance = _instance;
  init_info.PhysicalDevice = _chosenGPU;
  init_info.Device = _device;
  init_info.Queue = _graphicsQueue;
  init_info.DescriptorPool = imguiPool;
  init_info.MinImageCount = get_frame_overlay();
  init_info.ImageCount = get_frame_overlay();
  init_info.UseDynamicRendering = true;

  // dynamic rendering parameters for imgui to use
  init_info.PipelineRenderingCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  init_info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
  init_info.PipelineRenderingCreateInfo.pColorAttachmentFormats =
      &_swapchainImageFormat;

  init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

  ImGui_ImplVulkan_Init(&init_info);

  ImGui_ImplVulkan_CreateFontsTexture();

  // add the destroy the imgui created structures
  _mainDeletionQueue.push_function([=, this]() {
    ImGui_ImplVulkan_Shutdown();
    vkDestroyDescriptorPool(_device, imguiPool, nullptr);
  });
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

  VkCommandBufferSubmitInfo cmdinfo = vkinit::command_buffer_submit_info(cmd);
  VkSubmitInfo2 submit = vkinit::submit_info(&cmdinfo, nullptr, nullptr);

  // submit command buffer to the queue and execute it.
  //  _renderFence will now block until the graphic commands finish execution
  VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit, _immFence));

  VK_CHECK(vkWaitForFences(_device, 1, &_immFence, true, 9999999999));
}

void VulkanEngine::cleanup() {
  if (_isInitialized) {
    vkDeviceWaitIdle(_device);
    for (int i = 0; i < get_frame_overlay(); i++) {

      // already written from before
      vkDestroyCommandPool(_device, _frames[i]._commandPool, nullptr);

      // destroy sync objects
      vkDestroyFence(_device, _frames[i]._renderFence, nullptr);
      vkDestroySemaphore(_device, _frames[i]._renderSemaphore, nullptr);
      vkDestroySemaphore(_device, _frames[i]._swapchainSemaphore, nullptr);
      _frames[i]._deletionQueue.flush();
    }
    _mainDeletionQueue.flush();
    destory_swapchain();
    vkDestroySurfaceKHR(_instance, _surface, nullptr);
    vkDestroyDevice(_device, nullptr);

    vkb::destroy_debug_utils_messenger(_instance, _debug_messager);
    vkDestroyInstance(_instance, nullptr);
    SDL_DestroyWindow(_window);
  }

  loadedEngine = nullptr;
}

void VulkanEngine::draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView)
{
  VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(targetImageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  VkRenderingInfo renderInfo = vkinit::rendering_info(_swapchainExtent, &colorAttachment, nullptr);
  
  vkCmdBeginRendering(cmd, &renderInfo);
  
  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
  
  vkCmdEndRendering(cmd);
}

void VulkanEngine::draw_background(VkCommandBuffer cmd) {
  // make a clear-color from frame number. This will flash with a 120 frame
  // period.
  //  VkClearColorValue clearValue;
  //  float flash = std::abs(std::sin(_frameNumber / 120.f));
  //  clearValue = { { 0.0f, 0.0f, flash, 1.0f } };
  //
  //  VkImageSubresourceRange clearRange =
  //  vkinit::image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);
  //  //clear image
  //  vkCmdClearColorImage(cmd, _drawImage.image, VK_IMAGE_LAYOUT_GENERAL,
  //  &clearValue, 1, &clearRange);
  //


  ComputeEffect& effect = backgroundEffects[currentBackgroundEffect];
  // bind the gradient drawing compute pipeline
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, effect.pipeline);

  // bind the descriptor set containing the draw image for the compute pipeline
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          _gradientPipelineLayout, 0, 1, &_drawImageDescriptors,
                          0, nullptr);

  vkCmdPushConstants(cmd, _gradientPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputePushConstants), &effect.data);
  vkCmdDispatch(cmd, std::ceil(_drawExtent.width / 16.0), std::ceil(_drawExtent.height / 16.0), 1);



  // execute the compute pipeline dispatch. We are using 16x16 workgroup size so
  // we need to divide by it
  vkCmdDispatch(cmd, std::ceil(_drawExtent.width / 16.0),
                std::ceil(_drawExtent.height / 16.0), 1);
}

void VulkanEngine::draw() {
  VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._renderFence, true,
                           SecondsInNano(1)));
  VK_CHECK(vkResetFences(_device, 1, &get_current_frame()._renderFence));

  uint32_t swapchainImageIndex;
  VK_CHECK(vkAcquireNextImageKHR(_device, _swapchain, SecondsInNano(1),
                                 get_current_frame()._swapchainSemaphore,
                                 VK_NULL_HANDLE, &swapchainImageIndex));
  VkCommandBuffer cmd = get_current_frame()._mainCommandBuffer;
  VK_CHECK(vkResetCommandBuffer(cmd, 0));

  // begin the command buffer recording. We will use this command buffer exactly
  // once, so we want to let vulkan know that
  VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(
      VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

  _drawExtent.width = _windowExtent.width;
  _drawExtent.height = _windowExtent.height;

  // start the command buffer recording
  VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

  // make the swapchain image into writeable mode before rendering
  vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_UNDEFINED,
                           VK_IMAGE_LAYOUT_GENERAL);

  draw_background(cmd);

  // make the swapchain image into presentable mode
  vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_GENERAL,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
  vkutil::transition_image(cmd, _swapchainImage[swapchainImageIndex],
                           VK_IMAGE_LAYOUT_UNDEFINED,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  // execute a copy from the draw image into the swapchain
  vkutil::copy_image_to_image(cmd, _drawImage.image, _swapchainImage[swapchainImageIndex], _drawExtent, _swapchainExtent);

     // set swapchain image layout to Attachment Optimal so we can draw it
  vkutil::transition_image(cmd, _swapchainImage[swapchainImageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

  //draw imgui into the swapchain image
  draw_imgui(cmd,  _swapchainImageViews[swapchainImageIndex]);

     // set swapchain image layout to Present so we can draw it
  vkutil::transition_image(cmd, _swapchainImage[swapchainImageIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);


  // finalize the command buffer (we can no longer add commands, but it can now
  // be executed)
  VK_CHECK(vkEndCommandBuffer(cmd));

  VkCommandBufferSubmitInfo cmdinfo = vkinit::command_buffer_submit_info(cmd);

  VkSemaphoreSubmitInfo waitInfo = vkinit::semaphore_submit_info(
      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR,
      get_current_frame()._swapchainSemaphore);
  VkSemaphoreSubmitInfo signalInfo =
      vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
                                    get_current_frame()._renderSemaphore);

  VkSubmitInfo2 submit = vkinit::submit_info(&cmdinfo, &signalInfo, &waitInfo);

  // submit command buffer to the queue and execute it.
  //  _renderFence will now block until the graphic commands finish execution
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
      .pResults = nullptr,
  };

  VK_CHECK(vkQueuePresentKHR(_graphicsQueue, &presentInfo));

  // increase the number of frames drawn
  get_current_frame()._deletionQueue.push_function([&]() { _frameNumber++; });
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
      //send SDL event to imgui for handling
      ImGui_ImplSDL2_ProcessEvent(&e);
    }

    // do not draw if we are minimized
    if (stop_rendering) {
      // throttle the speed to avoid the endless spinning
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }

    // imgui new frame
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();

    if(ImGui::Begin("background")){
        ComputeEffect& selected = backgroundEffects[currentBackgroundEffect];
		
	ImGui::Text("Selected effect: ", selected.name);
	
	ImGui::SliderInt("Effect Index", &currentBackgroundEffect,0, backgroundEffects.size() - 1);
	
	ImGui::InputFloat4("data1",(float*)& selected.data.data1);
	ImGui::InputFloat4("data2",(float*)& selected.data.data2);
	ImGui::InputFloat4("data3",(float*)& selected.data.data3);
	ImGui::InputFloat4("data4",(float*)& selected.data.data4);
    }
    ImGui::End();


    //make imgui calculate internal draw structures
    ImGui::Render();

    draw();
  }
}
