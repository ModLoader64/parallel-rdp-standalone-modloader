#include "rdp_renderer.hpp"

namespace RDP {
namespace {
constexpr uint32_t gSlotMinimumSize = 64 * 1024;
constexpr uint64_t gSlotRetireScanouts = 120;

bool In_Block(uint32_t begin, uint32_t length, uint32_t size) {
    return (begin & ~(size - 1)) == ((begin + length - 1) & ~(size - 1));
}

bool Overlaps(uint32_t first, uint32_t first_length, uint32_t second, uint32_t second_length, uint32_t size) {
    return ((second - first) & (size - 1)) < first_length || ((first - second) & (size - 1)) < second_length;
}

bool Fits(uint32_t size, uint32_t color, uint32_t color_bytes, uint32_t border, uint32_t depth, uint32_t depth_bytes, bool has_depth) {
    if (color_bytes + 2 * border > size || !In_Block(color, color_bytes, size)) {
        return false;
    }
    if (!has_depth) {
        return true;
    }
    return depth_bytes <= size && In_Block(depth, depth_bytes, size) && !Overlaps(color - border, color_bytes + 2 * border, depth, depth_bytes, size);
}

uint32_t Pixel_Size_Log2(FBFormat format) {
    switch (format) {
    case FBFormat::RGBA8888:
        return 2;
    case FBFormat::RGBA5551:
    case FBFormat::IA88:
        return 1;
    default:
        return 0;
    }
}

void Mark_Drawn(uint32_t* begins, uint32_t* ends, uint64_t* stamps, uint32_t begin, uint32_t end, uint64_t stamp) {
    uint32_t oldest = 0;

    for (uint32_t index = 0; index < 4; index++) {
        if (begins[index] == begin || ends[index] == 0) {
            oldest = index;
            break;
        }

        if (stamps[index] < stamps[oldest]) {
            oldest = index;
        }
    }
    begins[oldest] = begin;
    ends[oldest] = end;
    stamps[oldest] = stamp;
}
} // namespace

uint32_t Renderer::get_upscaled_rdram_size() const {
    return scanout_slot.size;
}

void Renderer::reset_upscaled_slots() {
    for (auto& slot : upscaled_slots) {
        slot = {};
    }
    upscaled_slot = nullptr;
    scanout_slot = {};
}

Renderer::UpscaledSlot* Renderer::whole_rdram_slot() {
    UpscaledSlot* slot = &upscaled_slots[0];

    slot->size = uint32_t(rdram_size);
    slot->planes = upscaling_multisampled_rdram;
    slot->hidden = upscaling_multisampled_hidden_rdram;
    slot->reference = upscaling_reference_rdram;
    return slot;
}

bool Renderer::upscaled_slot_holds(const UpscaledSlot& slot, uint32_t begin, uint32_t length, uint32_t border) const {
    uint32_t block = begin & ~(slot.size - 1);

    if (slot.size == 0 || length + 2 * border > slot.size || !In_Block(begin, length, slot.size)) {
        return false;
    }

    if (slot.color_base == slot.depth_base) {
        return block == slot.color_base;
    }

    if (block == slot.color_base) {
        return !Overlaps(begin - border, length + 2 * border, slot.depth, slot.depth_bytes, slot.size);
    }

    return block == slot.depth_base && begin >= slot.depth && begin + length <= slot.depth + slot.depth_bytes;
}

Renderer::UpscaledSlot* Renderer::create_upscaled_slot(Vulkan::CommandBuffer& cmd, uint32_t size, uint32_t color_base, uint32_t depth_base, bool has_depth) {
    UpscaledSlot* slot = nullptr;
    Vulkan::BufferCreateInfo info = {};
    uint32_t factor = caps.upscaling * caps.upscaling;

    for (auto& candidate : upscaled_slots) {
        if (candidate.size == 0) {
            slot = &candidate;
            break;
        }

        if (slot == nullptr || candidate.last_use < slot->last_use) {
            slot = &candidate;
        }
    }

    *slot = {};
    info.domain = Vulkan::BufferDomain::Device;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    info.misc = Vulkan::BUFFER_MISC_ZERO_INITIALIZE_BIT;
    info.size = size;
    slot->reference = device->create_buffer(info);
    info.size = VkDeviceSize(size) * factor;
    slot->planes = device->create_buffer(info);
    info.size = VkDeviceSize(size / 2) * factor;
    info.misc = 0;
    slot->hidden = device->create_buffer(info);
    if (!slot->reference || !slot->planes || !slot->hidden) {
        *slot = {};
        return nullptr;
    }

    cmd.fill_buffer(*slot->hidden, 0x03030303);
    cmd.barrier(VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    slot->size = size;
    slot->color_base = color_base;
    slot->depth_base = has_depth ? depth_base : color_base;
    slot->last_use = upscaled_scanouts;
    return slot;
}

Renderer::UpscaledSlot* Renderer::upscaled_slot_for_pass() {
    uint32_t mask = uint32_t(rdram_size - 1);
    uint32_t pixel_size_log2 = Pixel_Size_Log2(fb.fmt);
    uint32_t pixels = fb.width * fb.deduced_height;
    uint32_t color = fb.addr & mask;
    uint32_t color_bytes = ((pixels << pixel_size_log2) + 3) & ~3u;
    uint32_t depth = fb.depth_addr & mask;
    uint32_t depth_bytes = (pixels * 2 + 3) & ~3u;
    uint32_t border = 3 * (fb.width << pixel_size_log2) + 16;
    bool has_depth = false;
    UpscaledSlot* slot = nullptr;
    uint64_t stamp = 0;

    upscaled_slot_depth = true;
    if (caps.upscaling <= 1) {
        return nullptr;
    }

    if (caps.super_sample_readback) {
        return whole_rdram_slot();
    }

    if (pixels == 0 || stream.span_info_jobs.empty() || !should_render_upscaled()) {
        return nullptr;
    }

    for (uint32_t index = 0; index < stream.depth_blend_state_cache.size(); index++) {
        has_depth |= (stream.depth_blend_state_cache.data()[index].flags & (DEPTH_BLEND_DEPTH_TEST_BIT | DEPTH_BLEND_DEPTH_UPDATE_BIT)) != 0;
    }

    has_depth = has_depth && fb.addr != fb.depth_addr;
    upscaled_slot_depth = has_depth;
    for (auto& candidate : upscaled_slots) {
        if (!upscaled_slot_holds(candidate, color, color_bytes, border) ||
            (has_depth && (!upscaled_slot_holds(candidate, depth, depth_bytes, 0) ||
                           Overlaps(color - border, color_bytes + 2 * border, depth, depth_bytes, candidate.size)))) {
            continue;
        }

        for (uint32_t index = 0; index < 4; index++) {
            if (slot == nullptr || candidate.drawn_at[index] > stamp) {
                slot = &candidate;
                stamp = candidate.drawn_at[index];
            }
        }
    }

    for (uint32_t size = gSlotMinimumSize; slot == nullptr && size <= rdram_size; size <<= 1) {
        if (Fits(size, color, color_bytes, border, depth, depth_bytes, has_depth)) {
            slot = create_upscaled_slot(*stream.cmd, size, color & ~(size - 1), depth & ~(size - 1), has_depth);
            if (slot == nullptr) {
                return nullptr;
            }

            slot->depth = has_depth ? depth : 0;
            slot->depth_bytes = has_depth ? depth_bytes : 0;
            for (auto& other : upscaled_slots) {
                bool held = &other != slot && other.size != 0 && (other.depth_bytes == 0 || upscaled_slot_holds(*slot, other.depth, other.depth_bytes, 0));

                for (uint32_t index = 0; held && index < 4; index++) {
                    held = other.drawn_end[index] == 0 ||
                           upscaled_slot_holds(*slot, other.drawn_begin[index], other.drawn_end[index] - other.drawn_begin[index], 0);
                }

                if (held) {
                    other = {};
                }
            }
        }
    }

    if (slot != nullptr) {
        Mark_Drawn(slot->drawn_begin, slot->drawn_end, slot->drawn_at, color, color + color_bytes, ++upscaled_draws);
        slot->last_use = upscaled_scanouts;
    }

    return slot;
}

const Renderer::UpscaledSlot& Renderer::upscaled_slot_for_scanout(Vulkan::CommandBuffer& cmd, uint32_t addr, uint32_t length) {
    std::lock_guard<std::mutex> holder{idle_lock};
    uint32_t mask = uint32_t(rdram_size - 1);
    uint32_t middle = (addr + length / 2) & mask;
    uint32_t border = length / 32;
    uint32_t begin = (addr + border) & mask;
    UpscaledSlot* slot = nullptr;
    uint64_t stamp = 0;

    if (caps.super_sample_readback) {
        scanout_slot = *whole_rdram_slot();
        return scanout_slot;
    }

    upscaled_scanouts++;
    retire_upscaled_slots();
    for (auto& candidate : upscaled_slots) {
        for (uint32_t index = 0; candidate.size != 0 && index < 4; index++) {
            if (middle >= candidate.drawn_begin[index] && middle < candidate.drawn_end[index] &&
                (slot == nullptr || candidate.drawn_at[index] > stamp)) {
                slot = &candidate;
                stamp = candidate.drawn_at[index];
            }
        }
    }

    for (auto& candidate : upscaled_slots) {
        if (slot == nullptr && upscaled_slot_holds(candidate, begin, length - 2 * border, 0)) {
            slot = &candidate;
        }
    }

    for (uint32_t size = gSlotMinimumSize; slot == nullptr && size <= rdram_size; size <<= 1) {
        if (length <= size && (In_Block(begin, length - 2 * border, size) || size == rdram_size)) {
            slot = create_upscaled_slot(cmd, size, begin & ~(size - 1), 0, false);
        }
    }

    if (slot != nullptr) {
        slot->last_use = upscaled_scanouts;
        scanout_slot = *slot;
    }
    return scanout_slot;
}

void Renderer::retire_upscaled_slots() {
    for (auto& slot : upscaled_slots) {
        if (slot.size != 0 && slot.last_use + gSlotRetireScanouts < upscaled_scanouts) {
            slot = {};
        }
    }
}

void Renderer::submit_update_upscaled_slot(Vulkan::CommandBuffer& cmd, ResolveStage stage, const UpscaledSlot& slot, unsigned addr, unsigned depth_addr,
                                           unsigned width, unsigned height, unsigned pixel_size_log2) {
    uint32_t mask = uint32_t(rdram_size - 1) & ~(slot.size - 1);
    uint32_t color_base = (addr + ((width * height) << pixel_size_log2) / 2) & mask;
    uint32_t depth_base = (depth_addr + width * height) & mask;

    if (addr == depth_addr || color_base == depth_base) {
        submit_update_upscaled_domain(cmd, stage, addr, depth_addr, width, height, pixel_size_log2, slot, color_base);
        return;
    }

    submit_update_upscaled_domain(cmd, stage, addr, addr, width, height, pixel_size_log2, slot, color_base);
    submit_update_upscaled_domain(cmd, stage, depth_addr, depth_addr, width, height, 1, slot, depth_base);
}
} // namespace RDP
