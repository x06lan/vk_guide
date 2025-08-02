// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include "vk_descriptors.h"
#include "vk_loader.h"
#include <vk_types.h>
struct ComputePushConstants {
  glm::vec4 data1;
  glm::vec4 data2;
  glm::vec4 data3;
  glm::vec4 data4;
};

struct ComputeEffect {
  const char *name;
  VkPipeline pipeline;
  VkPipelineLayout layout;
  ComputePushConstants data;
};

struct DeletionQueue {
  std::deque<std::function<void()>> deletors;
  void push_function(std::function<void()> &&function) {
    deletors.push_back(std::move(function));
  }
  void flush() {
    for (auto it = deletors.rbegin(); it != deletors.rend(); ++it) {
      (*it)();
    }
    deletors.clear();
  }
};

struct FrameData {
  VkCommandPool _commandPool;
  VkCommandBuffer _mainCommandBuffer;
  VkSemaphore _swapchainSemaphore, _renderSemaphore;
  VkFence _renderFence;
  DeletionQueue _deletionQueue;
  DescriptorAllocatorGrowable _frameDescriptor;
};
struct GPUSceneData {
  glm::mat4 view;
  glm::mat4 proj;
  glm::mat4 viewProj;
  glm::vec4 ambientColor;
  glm::vec4 sunlightDirection;
  glm::vec4 sunlightColor;
};

struct GLTFMetallic_Roughness {
  MaterialPipeline opaquePipeline;
  MaterialPipeline transparentPipeline;

  VkDescriptorSetLayout materialLayout;

  struct MaterialConstants {
    glm::vec4 colorFactors;
    glm::vec4 metal_rough_factors;

    // Padding
    glm::vec4 extra[14];
  };
  struct MaterialResources {
    AllocatedImage colorImage;
    VkSampler colorSampler;
    AllocatedImage metalRoughImage;
    VkSampler metalRoughSampler;
    VkBuffer dataBuffer;
    uint32_t dataBufferOffset;
  };

  DescriptorWriter writer;

  void build_pipelines(VulkanEngine *engine);
  void clear_resources(VkDevice device);

  MaterialInstance
  write_material(VkDevice device, MaterialPass pass,
                 const MaterialResources &resources,
                 DescriptorAllocatorGrowable &descriptorAllocator);
};
struct MeshNode : public Node {

  std::shared_ptr<MeshAsset> mesh;

  virtual void Draw(const glm::mat4 &topMatrix, DrawContext &ctx) override;
};

#define SecondsInNano(x) (x * 1000000000LL)

class VulkanEngine {
public:
  std::vector<ComputeEffect> backgroundEffects;
  int currentEffectIndex{0};
  std::vector<FrameData> _frames;
  inline FrameData &get_current_frame() {
    return _frames[_frameNumber % _frames.size()];
  }
  inline uint32_t get_frame_overlap() { return _frames.size(); }
  VkQueue _graphicsQueue;
  uint32_t _graphicsQueueFamily;

  bool _isInitialized{false};
  int _frameNumber{0};
  bool stop_rendering{false};
  VkExtent2D _windowExtent{800, 600};
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
  DeletionQueue _mainDeletionQueue;
  VmaAllocator _allocator;
  AllocatedImage _drawImage;
  AllocatedImage _depthImage;
  VkExtent2D _drawExtent;
  float rendrScale = 1.0f;
  DescriptorAllocatorGrowable _globalDescriptorAllocator;

  VkFence _immFence;
  VkCommandBuffer _immCommandBuffer;
  VkCommandPool _immCommandPool;

  VkDescriptorSet _drawImageDescriptors;
  VkDescriptorSetLayout _drawImageDescriptorLayout;

  VkPipeline _gradientPipeline;
  VkPipelineLayout _gradientPipelineLayout;

  VkDescriptorPool imguiPool;

  GPUSceneData sceneData;
  VkDescriptorSetLayout _gpuSceneDataDescriptorLayout;

  VkDescriptorSetLayout _singleImageDescriptorLayout;

  bool resize_requested = false;

  std::vector<std::shared_ptr<MeshAsset>> testMeshes;

  AllocatedImage _whiteImage;
  AllocatedImage _blackImage;
  AllocatedImage _greyImage;
  AllocatedImage _errorCheckImage;

  VkSampler _defaultSamplerLinear;
  VkSampler _defaultSamplerNearest;

  MaterialInstance defaultData;
  GLTFMetallic_Roughness metalRoughMaterial;

  DrawContext mainDrawContext;
  std::unordered_map<std::string, std::shared_ptr<MeshNode>> loadedNodes;

  struct SDL_Window *_window{nullptr};

  static VulkanEngine &Get();

  GPUMeshBuffer uploadMesh(std::span<uint32_t> indices,
                           std::span<Vertex> vertices);

  AllocatedImage create_image(VkExtent3D size, VkFormat format,
                              VkImageUsageFlags usage, bool mipmapped = false);
  AllocatedImage create_image(void *data, VkExtent3D size, VkFormat format,
                              VkImageUsageFlags usage, bool mipmapped = false);

  void update_scene();

  void destroy_image(const AllocatedImage &img);

  void immediate_submit(std::function<void(VkCommandBuffer cmd)> &&function);
  // initializes everything in the engine
  void init();

  // shuts down the engine
  void cleanup();

  // draw loop
  void draw();

  // run main loop
  void run();

private:
  AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage,
                                VmaMemoryUsage memoryUsage);
  void init_default_data();
  void init_imgui();
  void draw_imgui(VkCommandBuffer cmd, VkImageView image);
  void init_pipelines();
  void init_vulkan();
  void init_swapchain();
  void init_commands();
  void init_sync_structures();
  void init_descriptors();
  void init_background_pipeline();
  void resize_swapchain();
  void draw_background(VkCommandBuffer cmd);
  void draw_geometry(VkCommandBuffer cmd);
  void create_swapchain(uint32_t wigth, uint32_t height);
  void destroy_swapchain();

  void destroy_buffer(const AllocatedBuffer &buffer);
};
