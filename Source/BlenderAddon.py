import bpy
import array
import math

import taichi as ti
import numpy as np
from Math import *
import Camera

class CustomRenderEngine(bpy.types.RenderEngine):
    # These three members are used by blender to set up the
    # RenderEngine; define its internal name, visible name and capabilities.
    bl_idname = "CUSTOM"
    bl_label = "Custom"
    bl_use_preview = True

    # Init is called whenever a new render engine instance is created. Multiple
    # instances may exist at the same time, for example for a viewport and final
    # render.
    def __init__(self):
        print('calling CustomRenderEngine.__init__')

        self.scene_data = None
        self.draw_data = None
        self.cameraLens = Camera.CameraLens()
        self.cameraTrans = Camera.CameraTransform()

        ti.init(arch=ti.vulkan)

    # When the render engine instance is destroy, this is called. Clean up any
    # render engine data here, for example stopping running render threads.
    def __del__(self):
        pass

    # This is the method called by Blender for both final renders (F12) and
    # small preview for materials, world and lights.
    def render(self, depsgraph):
        print('calling CustomRenderEngine.render')

        scene = depsgraph.scene
        scale = scene.render.resolution_percentage / 100.0
        self.size_x = int(scene.render.resolution_x * scale)
        self.size_y = int(scene.render.resolution_y * scale)
        
        # Fill the render result with a flat color. The framebuffer is
        # defined as a list of pixels, each pixel itself being a list of
        # R,G,B,A values.
        backBuffer = ti.Vector.field(4, dtype=ti.f32, shape=(self.size_x, self.size_y))
        backBuffer.fill(ti.Vector([0, 1, 0, 1]))
        npArr = np.reshape(backBuffer.to_numpy(), (self.size_x * self.size_y, 4))
        rect = npArr.tolist()

        # pixel_count = self.size_x * self.size_y
        # rect = [color] * pixel_count

        # Here we write the pixel values to the RenderResult
        result = self.begin_result(0, 0, self.size_x, self.size_y)
        layer = result.layers[0].passes["Combined"]
        layer.rect = rect
        self.end_result(result)

    # For viewport renders, this method gets called once at the start and
    # whenever the scene or 3D viewport changes. This method is where data
    # should be read from Blender in the same thread. Typically a render
    # thread will be started to do the work while keeping Blender responsive.
    def view_update(self, context, depsgraph):
        print('calling CustomRenderEngine.view_update')

        # region = context.region
        # view3d = context.space_data
        # scene = depsgraph.scene

        # blenderCamera = depsgraph.scene.camera
        # blenderViewMat = context.region_data.view_matrix.inverted()
        # viewMat = Mat44f(blenderViewMat)
        # cameraPos = viewMat @ Vec4f([0, 0, 0, 1])
        # cameraUp = viewMat @ Vec4f([0, 0, 1, 0])
        # cameraDir = viewMat @ Vec4f([0, 1, 0, 0])

        # self.camLens = Camera.CameraLens(
        #                     fovVert=blenderCamera.angle_y * 180.0 / math.pi, 
        #                     resolution=Vec2i(region.width, region.height), 
        #                     focusDistance=10.0, aperture=0.0)
        
        # self.camTrans = Camera.CameraTransform(
        #                     pos=cameraPos,
        #                     up=cameraUp,
        #                     dir=cameraDir)

        # print(str(self.camTrans.mPos))

        # if not self.scene_data:
        #     # First time initialization
        #     self.scene_data = []
        #     first_time = True

        #     # Loop over all datablocks used in the scene.
        #     for datablock in depsgraph.ids:
        #         pass
        # else:
        #     first_time = False

        #     # Test which datablocks changed
        #     for update in depsgraph.updates:
        #         print("Datablock updated: ", update.id.name)

        #     # Test if any material was added, removed or changed.
        #     if depsgraph.id_type_updated('MATERIAL'):
        #         print("Materials updated")

        # # Loop over all object instances in the scene.
        # if first_time or depsgraph.id_type_updated('OBJECT'):
        #     for instance in depsgraph.object_instances:
        #         pass

    def update(self, data=None, depsgraph=None):
        print('calling CustomRenderEngine.update')

        scene = depsgraph.scene
        blenderCamera = scene.camera

        # https://blender.stackexchange.com/questions/130948/blender-api-get-current-location-and-rotation-of-camera-tracking-an-object
        pos, rot, scale = blenderCamera.matrix_world.decompose()
        rotMatB = rot.to_matrix()
        rotMat = ti.Matrix.cols([
            Vec3f(rotMatB.col[0]), 
            Vec3f(rotMatB.col[1]), 
            Vec3f(rotMatB.col[2])])
        cameraPos = Vec3f(pos)
        cameraUp = rotMat @ Vec3f([0, 0, 1])
        cameraDir = rotMat @ Vec3f([0, 1, 0])

        # https://blender.stackexchange.com/questions/80195/how-to-get-the-truly-width-and-height-of-frame-when-rendering-it-would-be-good
        renderScale = scene.render.resolution_percentage / 100.0
        resolution = Vec2i(scene.render.resolution_x * renderScale, scene.render.resolution_y * renderScale)

        self.camLens = Camera.CameraLens(
                            fovVert=blenderCamera.data.angle_y * 180.0 / math.pi, 
                            resolution=resolution, 
                            focusDistance=10.0, aperture=0.0)
        
        self.camTrans = Camera.CameraTransform(
                            pos=cameraPos,
                            up=cameraUp,
                            dir=cameraDir)

    def view_update(self, context, depsgraph):
        print('calling CustomRenderEngine.view_update')

    # For viewport renders, this method is called whenever Blender redraws
    # the 3D viewport. The renderer is expected to quickly draw the render
    # with OpenGL, and not perform other expensive work.
    # Blender will draw overlays for selection and editing on top of the
    # rendered image automatically.
    def view_draw(self, context, depsgraph):
        print('calling CustomRenderEngine.view_draw')

        region = context.region
        scene = depsgraph.scene

        # Get viewport dimensions
        dimensions = region.width, region.height

        # Bind shader that converts from scene linear to display space,
        gpu.state.blend_set('ALPHA_PREMULT')
        self.bind_display_space_shader(scene)

        self.unbind_display_space_shader()
        gpu.state.blend_set('NONE')

# RenderEngines also need to tell UI Panels that they are compatible with.
# We recommend to enable all panels marked as BLENDER_RENDER, and then
# exclude any panels that are replaced by custom panels registered by the
# render engine, or that are not supported.
def get_panels():
    exclude_panels = {
        'VIEWLAYER_PT_filter',
        'VIEWLAYER_PT_layer_passes',
    }

    panels = []
    for panel in bpy.types.Panel.__subclasses__():
        if hasattr(panel, 'COMPAT_ENGINES') and 'BLENDER_RENDER' in panel.COMPAT_ENGINES:
            if panel.__name__ not in exclude_panels:
                panels.append(panel)

    return panels


def register():
    # Register the RenderEngine
    bpy.utils.register_class(CustomRenderEngine)

    for panel in get_panels():
        panel.COMPAT_ENGINES.add('CUSTOM')


def unregister():
    bpy.utils.unregister_class(CustomRenderEngine)

    for panel in get_panels():
        if 'CUSTOM' in panel.COMPAT_ENGINES:
            panel.COMPAT_ENGINES.remove('CUSTOM')


if __name__ == "__main__":
    register()