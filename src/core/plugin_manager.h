/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef NIXL_SRC_CORE_PLUGIN_MANAGER_H
#define NIXL_SRC_CORE_PLUGIN_MANAGER_H

#include <filesystem>
#include <string>
#include <map>
#include <set>
#include <memory>
#include <vector>
#include <mutex>

#include "backend/backend_plugin.h"
#include "device/device_allocator_plugin.h"
#include "device/utility_services.h"
#include "nixl_types.h"
#include "telemetry/telemetry_plugin.h"
#include "tracing/trace_plugin.h"

// Forward declarations
class nixlBackendEngine;
struct nixlBackendInitParams;
class nixlPluginManagerTestPeer;

class nixlPluginHandle {
public:
    nixlPluginHandle(void *handle) : handle_(handle) {}

    virtual ~nixlPluginHandle() = default;

    virtual const char *
    getName() const = 0;
    virtual const char *
    getVersion() const = 0;

protected:
    void *handle_; // Handle to the dynamically loaded library
};

/**
 * This class represents a NIXL plugin and is used to create plugin instances. nixlPluginHandle
 * attributes are modified only in the constructor and destructor and remain unchanged during normal
 * operation, e.g., query operations and plugin instance creation. This allows using it in
 * multi-threading environments without lock protection.
 */
class nixlBackendPluginHandle : public nixlPluginHandle {
public:
    nixlBackendPluginHandle(void *handle, nixlBackendPlugin *plugin);
    ~nixlBackendPluginHandle();

    nixlBackendEngine* createEngine(const nixlBackendInitParams* init_params) const;
    void destroyEngine(nixlBackendEngine* engine) const;
    const char *
    getName() const override;
    const char *
    getVersion() const override;
    nixl_b_params_t getBackendOptions() const;
    nixl_mem_list_t getBackendMems() const;

private:
    nixlBackendPlugin *plugin_;
};

// Structure to hold static plugin info
struct nixlBackendStaticPluginInfo {
    std::string name;
    nixlStaticPluginCreatorFunc createFunc;
};

struct nixlTelemetryStaticPluginInfo {
    std::string name;
    nixlTelemetryStaticPluginCreatorFunc createFunc;
};

class nixlTelemetryPluginHandle : public nixlPluginHandle {
public:
    nixlTelemetryPluginHandle(void *handle, nixlTelemetryPlugin *plugin);
    ~nixlTelemetryPluginHandle();

    std::unique_ptr<nixlTelemetryExporter>
    createExporter(const nixlTelemetryExporterInitParams &init_params) const;
    const char *
    getName() const override;
    const char *
    getVersion() const override;

private:
    nixlTelemetryPlugin *plugin_;
};

class nixlTracePluginHandle : public nixlPluginHandle {
public:
    nixlTracePluginHandle(void *handle, nixlTracePlugin *plugin);
    ~nixlTracePluginHandle();

    std::unique_ptr<nixl::trace::TraceBackend>
    createBackend(const nixlTraceBackendInitParams &init_params) const;
    const char *
    getName() const override;
    const char *
    getVersion() const override;

private:
    nixlTracePlugin *plugin_;
};

class nixlDeviceAllocatorPluginHandle : public nixlPluginHandle {
public:
    nixlDeviceAllocatorPluginHandle(void *handle, nixlDeviceAllocatorPluginV1 *plugin);
    ~nixlDeviceAllocatorPluginHandle();

    [[nodiscard]] nixlDeviceAllocator *
    getAllocator() const noexcept;
    [[nodiscard]] nixlDeviceRuntime
    getRuntime() const noexcept;
    const char *
    getName() const override;
    const char *
    getVersion() const override;

private:
    friend class nixlPluginManager;

    [[nodiscard]] bool
    initializeAllocator() const noexcept;

    nixlDeviceAllocatorPluginV1 *plugin_;
    mutable nixlDeviceAllocator *allocator_ = nullptr;
};

typedef std::shared_ptr<const nixlPluginHandle> (
    *nixlPluginLoaderFunc)(void *handle, const std::string &plugin_path);

class nixlPluginManager : public nixlUtilityServices {
public:
    // Singleton instance accessor
    static nixlPluginManager& getInstance();

    // Delete copy constructor and assignment operator
    nixlPluginManager(const nixlPluginManager&) = delete;
    nixlPluginManager& operator=(const nixlPluginManager&) = delete;

    void
    discoverPluginsFromList(const std::string &filename);

    // Load a specific backend plugin
    std::shared_ptr<const nixlBackendPluginHandle>
    loadBackendPlugin(const nixl_backend_t &plugin_name);

    // Load a specific telemetry plugin
    std::shared_ptr<const nixlTelemetryPluginHandle>
    loadTelemetryPlugin(const nixl_telemetry_plugin_t &plugin_name);

    // Load a specific trace-backend plugin
    std::shared_ptr<const nixlTracePluginHandle>
    loadTracePlugin(const std::string &plugin_name);

    // Unload backend plugin
    void
    unloadBackendPluginForUnitTest(const nixl_backend_t &plugin_name);

    [[nodiscard]] nixl_status_t
    getBackendParams(const nixl_backend_t &type,
                     nixl_mem_list_t &mems,
                     nixl_b_params_t &params) const;

    // Get a backend plugin handle
    std::shared_ptr<const nixlBackendPluginHandle>
    getBackendPlugin(const nixl_backend_t &plugin_name);

    // Get a telemetry plugin handle
    std::shared_ptr<const nixlTelemetryPluginHandle>
    getTelemetryPlugin(const nixl_telemetry_plugin_t &plugin_name);

    // Get all loaded backend plugin names
    std::vector<nixl_backend_t>
    getLoadedBackendPluginNames();

    // Get all available backend plugin names (loaded + discovered on disk)
    std::vector<nixl_backend_t>
    getAvailBackendPluginNames();

    // Get all loaded telemetry plugin names
    std::vector<nixl_telemetry_plugin_t>
    getLoadedTelemetryPluginNames();

    // Add a plugin directory
    void
    addPluginDirectory(const std::string &directory);

    // Static Plugin Helpers
    const std::vector<nixlBackendStaticPluginInfo> &
    getBackendStaticPlugins();

    const std::vector<nixlTelemetryStaticPluginInfo> &
    getTelemetryStaticPlugins();

    [[nodiscard]] nixlDeviceAllocator &
    deviceAllocator() noexcept override;

    [[nodiscard]] static std::vector<nixlDeviceRuntime>
    deviceAllocatorProbeOrder(unsigned num_nvidia_gpus, unsigned num_amd_gpus);

private:
    friend class nixlPluginManagerTestPeer;

    std::map<nixl_backend_t, std::shared_ptr<const nixlBackendPluginHandle>>
        loaded_backend_plugins_;
    std::map<nixl_telemetry_plugin_t, std::shared_ptr<const nixlTelemetryPluginHandle>>
        loaded_telemetry_plugins_;
    std::map<std::string, std::shared_ptr<const nixlTracePluginHandle>> loaded_trace_plugins_;
    std::map<nixlDeviceRuntime, std::vector<std::string>> device_allocator_plugin_paths_;
    std::shared_ptr<const nixlDeviceAllocatorPluginHandle> device_allocator_handle_;
    nixlDeviceAllocator *selected_device_allocator_ = nullptr;
    // Plugins discovered on disk but not yet dlopen'd
    std::set<nixl_backend_t> discovered_backend_plugins_;
    // Explicit paths from the plugin list file (name -> .so path)
    std::map<nixl_backend_t, std::string> explicit_plugin_paths_;
    std::vector<std::string> plugin_dirs_;
    std::vector<nixlBackendStaticPluginInfo> backend_static_plugins_;
    std::vector<nixlTelemetryStaticPluginInfo> telemetry_static_plugins_;
    mutable std::mutex mutex_;

    [[nodiscard]] std::shared_ptr<const nixlBackendPluginHandle>
    loadBackendPluginImpl(const nixl_backend_t &plugin_name) const;

    void
    registerBuiltinPlugins();
    void
    registerBackendStaticPlugin(const std::string &name, nixlStaticPluginCreatorFunc creator);
    void
    registerTelemetryStaticPlugin(const std::string &name,
                                  nixlTelemetryStaticPluginCreatorFunc creator);

    // Search a directory for plugins
    void
    discoverPluginsFromDir(const std::filesystem::path &dirpath);

    // Discover helper functions
    void
    discoverBackendPlugin(const std::string &filename);

    void
    discoverTelemetryPlugin(const std::string &filename);

    void
    discoverTracePlugin(const std::string &filename);

    void
    discoverDeviceAllocatorPlugin(const std::filesystem::path &path);

    [[nodiscard]] std::shared_ptr<const nixlDeviceAllocatorPluginHandle>
    loadDeviceAllocatorPlugin(nixlDeviceRuntime runtime) const;

    [[nodiscard]] static std::shared_ptr<const nixlDeviceAllocatorPluginHandle>
    loadDeviceAllocatorPluginFromPath(const std::string &path, nixlDeviceRuntime runtime);

    [[nodiscard]] static std::shared_ptr<const nixlPluginHandle>
    loadPluginFromPath(const std::string &plugin_path,
                       nixlPluginLoaderFunc loader,
                       bool deepbind = false);

    [[nodiscard]] static std::string
    composePluginPath(const std::string &dir,
                      const std::string &plugin_prefix,
                      const std::string &plugin_name);

    // Private constructor for singleton pattern
    nixlPluginManager();
};

#endif
