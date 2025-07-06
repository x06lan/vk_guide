// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vector>
#include <vk_types.h>
#include <vk_descriptors.h>


struct DeletionQueue
{
  std::deque<std::function<void()>> queue;
  void push_function(std::function<void()> func) { queue.push_back(func); }
  void flush()
  {
    for(auto it = queue.rbegin(); it != queue.rend(); ++it)
    {
      (*it)();
    }
  }
};
struct AllocatedImage {
    VkImage image;
    VkImageView imageView;
    VmaAllocation allocation;
    VkExtent3D imageExtent;
    VkFormat imageFormat;
};

struct FrameData
{
  VkCommandPool _commandPool;
  VkCommandBuffer _mainCommandBuffer;
  VkSemaphore _swapchainSemaphore, _renderSemaphore;
  VkFence _renderFence;
  DeletionQueue _deletionQueue;
};

#define SecondsInNano(x) (x * 1000000000LL)

class VulkanEngine
{
public:

  std::vector<FrameData> _frames;
  inline FrameData &get_current_frame() { return _frames[_frameNumber % _frames.size()]; }
  inline int get_frame_overlay() { return _frames.size(); }
  DeletionQueue _mainDeletionQueue;
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

  VmaAllocator _allocator;
  AllocatedImage _drawImage;
  VkExtent2D _drawExtent;

  DescriptorAllocator globalDescriptorAllocator;
  VkDescriptorSet _drawImageDescriptors;
  VkDescriptorSetLayout _drawImageDescriptorLayout;

  VkPipeline _gradientPipeline;
  VkPipelineLayout _gradientPipelineLayout;


  struct SDL_Window *_window{nullptr};

  static VulkanEngine &Get();

  // initializes everything in the engine
  void init();

  // shuts down the engine
  void cleanup();

  // draw loop
  void draw();
  void draw_background(VkCommandBuffer cmd);

  // run main loop
  void run();

private:
  void init_vulkan();
  void init_swapchain();
  void init_commands();
  void init_sync_structures();
  void create_swapchain(uint32_t wigth, uint32_t height);
  void destory_swapchain();
  void init_descriptors();
  void init_pipelines();
  void init_background_pipelines();

};
