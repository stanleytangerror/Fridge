import taichi as ti
import math
from pyquaternion import Quaternion
from Math import *

class CameraTransform:
    def __init__(self,  pos: Vec3f = Vec3f([0.0, 0.0, 0.0]), 
                        dir: Vec3f = Vec3f([0.0, 1.0, 0.0]), 
                        up: Vec3f = Vec3f([0.0, 0.0, 1.0])):
        assert abs(dir.dot(up)) < EPS
        self.mDir = dir.normalized()
        self.mUp = up.normalized()
        self.mRight = dir.cross(up)
        assert self.mRight.norm() == 1.0
        self.mPos = pos
    
class CameraLens:
    def __init__(self, fovVert, resolution: Vec2i, focusDistance, aperture):
        assert fovVert < 180.0
        self.mFovVer = fovVert / 180.0 * math.pi
        self.mResolution = resolution
        self.mAperture = aperture
        self.mFocusDistance = focusDistance

    @property
    def AspectRatio(self):
        return float(self.mResolution[0]) / float(self.mResolution[1])

    @property
    def InvResolution(self):
        return Vec2f([1.0 / self.mResolution[0], 1.0 / self.mResolution[1]])

class CameraControl:
    def __init__(self):
        self.mDraggingLastPos = False

    def UpdateCamera(self, window, cameraTrans, speed = 1.0):
        updated = False

        moveSpeed = 0.1 * speed
        keyMap = [
            ('w', cameraTrans.mDir),
            ('a', -cameraTrans.mRight),
            ('s', -cameraTrans.mDir),
            ('d', cameraTrans.mRight),
            ('e', cameraTrans.mUp),
            ('q', -cameraTrans.mUp)
        ]
        dir = Vec3f([0.0, 0.0, 0.0])
        for key, d in keyMap:
            if window.is_pressed(key): 
                dir = dir + d
                updated = True
        cameraTrans.mPos = cameraTrans.mPos + dir * moveSpeed

        rotSpeed = 0.02
        deltaPos = Vec2f([0.0, 0.0])
        if not window.is_pressed(ti.ui.LMB):
            self.mDraggingLastPos = None
        else:
            curPos = Vec2f(window.get_cursor_pos())
            if self.mDraggingLastPos is None:
                self.mDraggingLastPos = curPos
            deltaPos = curPos - self.mDraggingLastPos
            self.mDraggingLastPos = curPos
        
        if deltaPos.norm_sqr() * 1000.0 > EPS:
            updated = True
            v0 = cameraTrans.mDir
            v1 = -deltaPos[0] * cameraTrans.mRight - deltaPos[1] * cameraTrans.mUp
            axis = v1.cross(v0).normalized()
            rot = Quaternion(axis=axis, angle=rotSpeed)
            cameraTrans.mDir = Vec3f(rot.rotate(cameraTrans.mDir))
            cameraTrans.mUp = Vec3f(rot.rotate(cameraTrans.mUp))
            cameraTrans.mRight = Vec3f(rot.rotate(cameraTrans.mRight))

        return updated


@ti.data_oriented
class CameraDataConstBuffer:
    def __init__(self):
        self.cameraPos = Vec3f.field(shape=())
        self.cameraDir = Vec3f.field(shape=())
        self.up = Vec3f.field(shape=())
        self.right = Vec3f.field(shape=())
        self.fov = ti.field(dtype=ti.f32, shape=())
        self.aspectRatio = ti.field(dtype=ti.f32, shape=())
        self.invResolution = Vec2f.field(shape=())
        self.focusDist = ti.field(dtype=ti.f32, shape=())
        self.aperture = ti.field(dtype=ti.f32, shape=())

    def FillData(self, camLens: CameraLens, camTrans: CameraTransform):
        self.aspectRatio[None] = camLens.AspectRatio
        self.invResolution[None] = camLens.InvResolution
        self.focusDist[None] = camLens.mFocusDistance
        self.aperture[None] = camLens.mAperture
        self.fov[None] = camLens.mFovVer
        self.cameraPos[None] = camTrans.mPos
        self.cameraDir[None] = camTrans.mDir
        self.up[None] = camTrans.mUp
        self.right[None] = camTrans.mRight

@ti.func
def CastRayForPixel(u, v, cameraData):
    offset = cameraData.aperture[None] * 0.5 * RandomUnitDisk()
    o = cameraData.cameraPos[None] + offset[0] * cameraData.right[None] + offset[1] * cameraData.up[None]
    u = u + ti.random(ti.f32)
    v = v + ti.random(ti.f32)
    fu = 0.5 * cameraData.fov[None] * (u * cameraData.invResolution[None][0] - 0.5) * cameraData.aspectRatio[None]
    fv = 0.5 * cameraData.fov[None] * (v * cameraData.invResolution[None][1] - 0.5)
    disp = cameraData.focusDist[None] * Vec3f([ fu, fv, 1.0 ])
        
    d = (cameraData.cameraPos[None] + disp[2] * cameraData.cameraDir[None] + disp[0] * cameraData.right[None] + disp[1] * cameraData.up[None] - o).normalized()
    return o, d
