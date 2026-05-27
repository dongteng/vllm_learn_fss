# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from collections.abc import Callable
from typing import Any

import vllm.envs as envs

logger = logging.getLogger(__name__)

# Default plugins group will be loaded in all processes(process0, engine core
# process and worker processes)
DEFAULT_PLUGINS_GROUP = "vllm.general_plugins"
# IO processor plugins group will be loaded in process0 only
IO_PROCESSOR_PLUGINS_GROUP = "vllm.io_processor_plugins"
# Platform plugins group will be loaded in all processes when
# `vllm.platforms.current_platform` is called and the value not initialized,
PLATFORM_PLUGINS_GROUP = "vllm.platform_plugins"
# Stat logger plugins group will be loaded in process0 only when serve vLLM with
# async mode.
STAT_LOGGER_PLUGINS_GROUP = "vllm.stat_logger_plugins"

# make sure one process only loads plugins once
plugins_loaded = False


def load_plugins_by_group(group: str) -> dict[str, Callable[[], Any]]:
    from importlib.metadata import entry_points                                                                             #用于读取python包安装时注册的entry_points

    allowed_plugins = envs.VLLM_PLUGINS

    discovered_plugins = entry_points(group=group)
    if len(discovered_plugins) == 0:                                                                                        #没发现插件
        logger.debug("No plugins for group %s found.", group)
        return {}

    # Check if the only discovered plugin is the default one                                                                #判断是否是默认组
    is_default_group = group == DEFAULT_PLUGINS_GROUP
    # Use INFO for non-default groups and DEBUG for the default group
    log_level = logger.debug if is_default_group else logger.info

    log_level("Available plugins for group %s:", group)
    for plugin in discovered_plugins:
        log_level("- %s -> %s", plugin.name, plugin.value)

    if allowed_plugins is None:
        log_level(
            "All plugins in this group will be loaded. "
            "Set `VLLM_PLUGINS` to control which plugins to load."
        )

    plugins = dict[str, Callable[[], Any]]()
    for plugin in discovered_plugins:
        if allowed_plugins is None or plugin.name in allowed_plugins:
            if allowed_plugins is not None:
                log_level("Loading plugin %s", plugin.name)

            try:
                func = plugin.load()
                plugins[plugin.name] = func
            except Exception:
                logger.exception("Failed to load plugin %s", plugin.name)

    return plugins


def load_general_plugins():
    """WARNING: plugins can be loaded for multiple times in different                                                          警告:插件可能会在不同进程中被加载多次
    processes. They should be designed in a way that they can be loaded                                                        因此插件的设计必须保证:即使被重复加载多次,也不会产生问题
    multiple times without causing issues.
    """
    global plugins_loaded                                                                                                      #使用全局变量记录 当前进程是否已经加载过插件
    if plugins_loaded:
        return                                                                                                                 #如果加载过就直接返回
    plugins_loaded = True

    plugins = load_plugins_by_group(group=DEFAULT_PLUGINS_GROUP)                                                               #根据插件组加载插件                                                
    # general plugins, we only need to execute the loaded functions
    for func in plugins.values():
        func()
