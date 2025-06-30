// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>

struct FrameData
{
  VkCommandPool _commandPool;
  VkCommandBuffer _mainCommandBuffer;
  VkSemaphore _swapchainSemaphore, _renderSemaphore;
  VkFence _renderFence;
};

unsigned int constexpr FRAME_OVERLAP = 4;
#define SecondsInNano(x) (x * 1000000000LL)

class VulkanEngine
{
public:
  FrameData _frames[FRAME_OVERLAP];
  FrameData &get_current_frame() { return _frames[_frameNumber % FRAME_OVERLAP]; }
  VkQueue _graphicsQueue;
  uint32_t _graphicsQueueFamily;

  bool _isInitialized{false};
  int _frameNumber{0};
  bool stop_rendering{false};
  VkExtent2D _windowExtent{1700, 900};
  VkInstance _instance;
  VkDebugUtilsMessengerEXT _debug_messager;
  VkPhysicalDevice _chosenGPU;
  VkDevice _device;
  VkSurfaceKHR _surface;
  VkSwapchainKHR _swapchain;
  VkFormat _swapchainImageFormat;

  std::vector<VkImage> _swapchainImage;
  std::vector<VkImageView> _swapchainImageViews;
  VkExtent2D _swapchainExtent;

  struct SDL_Window *_window{nullptr};

  static VulkanEngine &Get();

  // initializes everything in the engine
  void init();

  // shuts down the engine
  void cleanup();

  // draw loop
  void draw();

  // run main loop
  void run();

private:
  void init_vulkan();
  void init_swapchain();
  void init_commands();
  void init_sync_structures();
  void create_swapchain(uint32_t wigth, uint32_t height);
  void destory_swapchain();
};
