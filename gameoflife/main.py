# SPDX-License-Identifier: Apache-2.0

from time import time
import sgl
from pathlib import Path
import numpy as np

EXAMPLE_DIR = Path(__file__).parent


class App:
    def __init__(self):
        super().__init__()
        self.window = sgl.Window(
            width=1920, height=1280, title="Game Of Life", resizable=True
        )
        self.device = sgl.Device(
            enable_debug_layers=True,
            compiler_options={"include_paths": [EXAMPLE_DIR]},
        )
        self.swapchain = self.device.create_swapchain(
            image_count=3,
            width=self.window.width,
            height=self.window.height,
            window=self.window,
            enable_vsync=False,
        )

        self.framebuffers = []
        self.create_framebuffers()

        self.ui = sgl.ui.Context(self.device)

        self.output_texture = None

        program = self.device.load_program("draw", ["main"])
        self.drawkernel = self.device.create_compute_kernel(program)

        program = self.device.load_program("draw", ["update"])
        self.updatekernel = self.device.create_compute_kernel(program)

        self.game_dimensions = sgl.int2(1024,1024)

        self.game_buffers = [
            self.device.create_buffer(
                element_count=self.game_dimensions.x*self.game_dimensions.y, 
                struct_size=4,
                usage=sgl.ResourceUsage.shader_resource | sgl.ResourceUsage.unordered_access,
                data=np.random.randint(0,100,(self.game_dimensions.x,self.game_dimensions.y))
                )
            for x in range(0,2)
        ]
        self.read_idx = 0

        self.offset = sgl.float2(0,0)
        self.scale = 2.0

        self.mouse_pos = sgl.float2()
        self.mouse_down = False
        self.playing = True

        self.window.on_keyboard_event = self.on_keyboard_event
        self.window.on_mouse_event = self.on_mouse_event
        self.window.on_resize = self.on_resize

        self.setup_ui()

    def setup_ui(self):
        screen = self.ui.screen
        window = sgl.ui.Window(screen, "Settings", size=sgl.float2(500, 300))

        self.fps_text = sgl.ui.Text(window, "FPS: 0")

        def start():
            self.playing = True

        sgl.ui.Button(window, "Start", callback=start)

        def stop():
            self.playing = False

        sgl.ui.Button(window, "Stop", callback=stop)


    def on_keyboard_event(self, event: sgl.KeyboardEvent):
        if self.ui.handle_keyboard_event(event):
            return

        if event.type == sgl.KeyboardEventType.key_press:
            if event.key == sgl.KeyCode.escape:
                self.window.close()

    def on_mouse_event(self, event: sgl.MouseEvent):
        if self.ui.handle_mouse_event(event):
            return

        if event.type == sgl.MouseEventType.move:
            self.mouse_pos = event.pos
        elif event.type == sgl.MouseEventType.button_down:
            if event.button == sgl.MouseButton.left:
                self.mouse_down = True
        elif event.type == sgl.MouseEventType.button_up:
            if event.button == sgl.MouseButton.left:
                self.mouse_down = False
        elif event.type == sgl.MouseEventType.scroll:
            print(event.scroll)
            self.scale += event.scroll.y*0.1

    def window_2_game_coord(self, win_coord: sgl.float2):
        coord = sgl.float2(win_coord)
        coord -= sgl.float2(self.output_texture.width,self.output_texture.height)*0.5
        coord /= self.scale
        coord += self.offset
        coord += sgl.float2(self.game_dimensions) * 0.5
        return coord

    def on_resize(self, width: int, height: int):
        self.framebuffers.clear()
        self.device.wait()
        self.swapchain.resize(width, height)
        self.create_framebuffers()

    def create_framebuffers(self):
        self.framebuffers = [
            self.device.create_framebuffer(render_targets=[image.get_rtv()])
            for image in self.swapchain.images
        ]

    def run(self):

        last_step = time()

        while not self.window.should_close():
            self.window.process_events()
            self.ui.process_events()

            image_index = self.swapchain.acquire_next_image()
            if image_index < 0:
                continue

            image = self.swapchain.get_image(image_index)
            if (
                self.output_texture == None
                or self.output_texture.width != image.width
                or self.output_texture.height != image.height
            ):
                self.output_texture = self.device.create_texture(
                    format=sgl.Format.rgba16_float,
                    width=image.width,
                    height=image.height,
                    mip_count=1,
                    usage=sgl.ResourceUsage.shader_resource
                    | sgl.ResourceUsage.unordered_access,
                    debug_name="output_texture",
                )
            
            if self.mouse_down:
                buff = self.game_buffers[self.read_idx].to_numpy().view(dtype=np.int32)
                fpcoord = self.window_2_game_coord(self.mouse_pos)
                coord = sgl.int2(int(fpcoord.x),int(fpcoord.y))
                for x in range(-1,2):
                    for y in range(-1,2):
                        idx = self.game_dimensions.x*(coord.y+y)+coord.x+x
                        if idx >= 0 and idx < len(buff):
                            buff[idx] = 1
                self.game_buffers[self.read_idx].from_numpy(buff)

            new_time = time()
            if (new_time-last_step) > 1/60.0:
                self.updatekernel.dispatch(
                    thread_count=[self.game_dimensions.x, self.game_dimensions.y, 1],
                    vars={
                        "g_game_dimensions": self.game_dimensions,
                        "g_game_in": self.game_buffers[self.read_idx],
                        "g_game_out": self.game_buffers[1-self.read_idx]
                    }
                )
                self.read_idx = 1 - self.read_idx
                last_step = new_time

            command_buffer = self.device.create_command_buffer()
            self.drawkernel.dispatch(
                thread_count=[self.output_texture.width, self.output_texture.height, 1],
                vars={
                    "g_game_dimensions": self.game_dimensions,
                    "g_game_in": self.game_buffers[self.read_idx],
                    "g_game_out": self.game_buffers[1-self.read_idx],
                    "g_draw_offset": self.offset,
                    "g_draw_scale": self.scale,
                    "g_output": self.output_texture,
                },
                command_buffer=command_buffer,
            )
            command_buffer.blit(image, self.output_texture)

            self.ui.new_frame(image.width, image.height)
            self.ui.render(self.framebuffers[image_index], command_buffer)

            command_buffer.set_texture_state(image, sgl.ResourceState.present)
            command_buffer.submit()
            del image

            self.swapchain.present()
            self.device.run_garbage_collection()


if __name__ == "__main__":
    app = App()
    app.run()