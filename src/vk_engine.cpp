//> includes
#include "vk_engine.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <vk_initializers.h>
#include <vk_types.h>
#include <vk_images.h>

#include "VkBootstrap.h"
#include <chrono>
#include <thread>
#include <vulkan/vulkan_core.h>

namespace
{
  VulkanEngine *loadedEngine = nullptr;
  auto constexpr bUseValidationLayers = true;
}; // namespace

VulkanEngine &VulkanEngine::Get() { return *loadedEngine; }

void VulkanEngine::init()
{
  // only one engine initialization is allowed with the application.
  assert(loadedEngine == nullptr);
  loadedEngine = this;

  // We initialize SDL and create a window with it.
  SDL_Init(SDL_INIT_VIDEO);

  SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);

  _window = SDL_CreateWindow("Vulkan Engine", SDL_WINDOWPOS_UNDEFINED,
                             SDL_WINDOWPOS_UNDEFINED, _windowExtent.width,
                             _windowExtent.height, window_flags);
  if (!_window)
  {
    fmt::println(stderr, "Can't Create window, SDLError: {}", SDL_GetError());
    std::abort();
  }

  init_vulkan();
  init_swapchain();
  init_commands();
  init_sync_structures();

  // everything went fine
  _isInitialized = true;
}

void VulkanEngine::init_vulkan()
{
  vkb::InstanceBuilder builder;
  auto const inst_ret = builder.set_app_name("Example Vulkan Application")
                            .request_validation_layers(bUseValidationLayers)
                            .use_default_debug_messenger()
                            .require_api_version(1, 3, 0)
                            .build();

  RESULT_CHECK(inst_ret, "Can't Init Vulkan Instance, Error: {}");
  _instance = inst_ret->instance;
  _debug_messager = inst_ret->debug_messenger;

  if (SDL_Vulkan_CreateSurface(_window, _instance, &_surface) != SDL_TRUE)
  {
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
}

void VulkanEngine::create_swapchain(uint32_t width, uint32_t height)
{
  vkb::SwapchainBuilder swapchainBuilder{_chosenGPU, _device, _surface};
  _swapchainImageFormat = VK_FORMAT_B8G8R8A8_UNORM;
  vkb::Result<vkb::Swapchain> vkbSwapchain =
      swapchainBuilder
          .set_desired_format(VkSurfaceFormatKHR{
              .format = _swapchainImageFormat,
              .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
          .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
          .set_desired_extent(width, height)
          .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
          .build();

  RESULT_CHECK(vkbSwapchain, "Failed to create swapchain, errormessage: {}");
  _swapchainExtent = vkbSwapchain->extent;
  _swapchain = vkbSwapchain->swapchain;
  fmt::println("Swapchain created with extent: {}x{}",
               _swapchainExtent.width, _swapchainExtent.height);

  auto images = vkbSwapchain->get_images();
  RESULT_CHECK(images, "Failed to get image, errormessage: {}");
  _swapchainImage = *images;

  auto image_views = vkbSwapchain->get_image_views();

  RESULT_CHECK(image_views, "Failed to get image view, errormessage: {}");

  _swapchainImageViews = *image_views;
}

void VulkanEngine::init_swapchain()
{
  create_swapchain(_windowExtent.width, _windowExtent.height);
}

void VulkanEngine::destory_swapchain()
{
  vkDestroySwapchainKHR(_device, _swapchain, nullptr);

  for (auto &imgView : _swapchainImageViews)
  {
    vkDestroyImageView(_device, imgView, nullptr);
  }
}

void VulkanEngine::init_commands()
{
  VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(
      _graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

  for (int i = 0; i < FRAME_OVERLAP; i++) // Start from i = 0
  {
    VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr,
                                 &_frames[i]._commandPool));

    VkCommandBufferAllocateInfo cmdAllocInfo =
        vkinit::command_buffer_allocate_info(_frames[i]._commandPool, 1);

    VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo,
                                      &_frames[i]._mainCommandBuffer));
  }
}

void VulkanEngine::init_sync_structures()
{
  VkFenceCreateInfo fenceInfo =
      vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
  VkSemaphoreCreateInfo semaphoreInfo = vkinit::semaphore_create_info();
  for (int i = 0; i < FRAME_OVERLAP; i++)
  {
    VK_CHECK(
        vkCreateFence(_device, &fenceInfo, nullptr, &_frames[i]._renderFence));
    VK_CHECK(vkCreateSemaphore(_device, &semaphoreInfo, nullptr,
                               &_frames[i]._renderSemaphore));
    VK_CHECK(vkCreateSemaphore(_device, &semaphoreInfo, nullptr,
                               &_frames[i]._swapchainSemaphore));
  }
}

void VulkanEngine::cleanup()
{
  if (_isInitialized)
  {
    vkDeviceWaitIdle(_device);
    for (int i = 0; i < FRAME_OVERLAP; i++)
    {

      // already written from before
      vkDestroyCommandPool(_device, _frames[i]._commandPool, nullptr);

      // destroy sync objects
      vkDestroyFence(_device, _frames[i]._renderFence, nullptr);
      vkDestroySemaphore(_device, _frames[i]._renderSemaphore, nullptr);
      vkDestroySemaphore(_device, _frames[i]._swapchainSemaphore, nullptr);
    }
    destory_swapchain();
    vkDestroySurfaceKHR(_instance, _surface, nullptr);
    vkDestroyDevice(_device, nullptr);

    vkb::destroy_debug_utils_messenger(_instance, _debug_messager);
    vkDestroyInstance(_instance, nullptr);
    SDL_DestroyWindow(_window);
  }
  loadedEngine = nullptr;
}

void VulkanEngine::draw()
{
  VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._renderFence, true,
                           SecondsInNano(1)));
  VK_CHECK(vkResetFences(_device, 1, &get_current_frame()._renderFence));

  uint32_t swapchainImageIndex;
  VK_CHECK(vkAcquireNextImageKHR(_device, _swapchain, SecondsInNano(1),
                                 get_current_frame()._swapchainSemaphore,
                                 VK_NULL_HANDLE, &swapchainImageIndex));
  VkCommandBuffer cmd = get_current_frame()._mainCommandBuffer;
  VK_CHECK(vkResetCommandBuffer(cmd, 0));

  // begin the command buffer recording. We will use this command buffer exactly once, so we want to let vulkan know that
  VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

  // start the command buffer recording
  VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

  // make the swapchain image into writeable mode before rendering
  vkutil::transition_image(cmd, _swapchainImage[swapchainImageIndex], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

  // make a clear-color from frame number. This will flash with a 120 frame period.
  VkClearColorValue clearValue;
  float flash = std::abs(std::sin(_frameNumber / 120.f));
  clearValue = {{0.0f, 0.0f, flash, 1.0f}};

  VkImageSubresourceRange clearRange = vkinit::image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);

  // clear image
  vkCmdClearColorImage(cmd, _swapchainImage[swapchainImageIndex], VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &clearRange);

  // make the swapchain image into presentable mode
  vkutil::transition_image(cmd, _swapchainImage[swapchainImageIndex], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

  // finalize the command buffer (we can no longer add commands, but it can now be executed)
  VK_CHECK(vkEndCommandBuffer(cmd));

  VkCommandBufferSubmitInfo cmdinfo = vkinit::command_buffer_submit_info(cmd);

  VkSemaphoreSubmitInfo waitInfo = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR, get_current_frame()._swapchainSemaphore);
  VkSemaphoreSubmitInfo signalInfo = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, get_current_frame()._renderSemaphore);

  VkSubmitInfo2 submit = vkinit::submit_info(&cmdinfo, &signalInfo, &waitInfo);

  // submit command buffer to the queue and execute it.
  //  _renderFence will now block until the graphic commands finish execution
  VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit, get_current_frame()._renderFence));

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
  _frameNumber++;
}

void VulkanEngine::run()
{
  SDL_Event e;
  bool bQuit = false;

  // main loop
  while (!bQuit)
  {
    // Handle events on queue
    while (SDL_PollEvent(&e) != 0)
    {
      // close the window when user alt-f4s or clicks the X button
      if (e.type == SDL_QUIT)
        bQuit = true;

      if (e.type == SDL_WINDOWEVENT)
      {
        if (e.window.event == SDL_WINDOWEVENT_MINIMIZED)
        {
          stop_rendering = true;
        }
        if (e.window.event == SDL_WINDOWEVENT_RESTORED)
        {
          stop_rendering = false;
        }
      }
    }

    // do not draw if we are minimized
    if (stop_rendering)
    {
      // throttle the speed to avoid the endless spinning
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }

    draw();
  }
}
