import imp
from pydoc import importfile
from re import T
import taichi as ti
from taichi.math import *
import time
import numpy as np
from Math import *
from GradientNoise import *
from Camera import *

ti.init(arch=ti.vulkan)

TRay = ti.types.struct(origin=Vec3f, direction=Vec3f)

depth = 10

Zero3f = Vec3f([0.0, 0.0, 0.0])
One3f = Vec3f([1.0, 1.0, 1.0])
XUnit3f = Vec3f([1.0, 0.0, 0.0])
YUnit3f = Vec3f([0.0, 1.0, 0.0])
ZUnit3f = Vec3f([0.0, 0.0, 1.0])

WindowSize = (1920, 1080)
# WindowSize = (32, 18)
window = ti.ui.Window("Smoke", WindowSize, vsync=True)

windowImage = ti.Vector.field(3, float, shape=WindowSize)
debugBuffer = ti.field(dtype=ti.f32, shape=WindowSize)
sceneColorBuffer = Vec3f.field(shape=WindowSize)

elapsedTime = ti.field(dtype=ti.f32, shape=())
beginTime = time.time()
noise = GradientNoise()

cameraTrans = CameraTransform(Vec3f([0.0, -10.0, 0.0]))
lens = CameraLens(90.0, WindowSize, 10, 0.0)
cameraControl = CameraControl()

cameraPos = Vec3f.field(shape=())
cameraDir = Vec3f.field(shape=())
up = Vec3f.field(shape=())
right = Vec3f.field(shape=())
fov = ti.field(dtype=ti.f32, shape=())
accumulate = ti.field(dtype=ti.f32, shape=())
aspectRatio = lens.AspectRatio
invResolution = lens.InvResolution
focusDist = lens.mFocusDistance
aperture = lens.mAperture

@ti.func
def CastRay(u, v):
    u = u + ti.random(ti.f32)
    v = v + ti.random(ti.f32)
    fu = 0.5 * fov[None] * (u * invResolution[0] - 0.5) * aspectRatio
    fv = 0.5 * fov[None] * (v * invResolution[1] - 0.5)
    disp = focusDist * Vec3f([ fu, fv, 1.0 ])
        
    d = (disp[2] * cameraDir[None] + disp[0] * right[None] + disp[1] * up[None]).normalized()

    return TRay(origin=cameraPos[None], direction=d)

@ti.func
def HitGround(ray, maxDist):
    dist = maxDist
    hit = False
    normal = ZUnit3f

    if ray.direction[2] < 0.0:
        dist = ti.min(dist, (ray.origin[2] + 3.0) / (-ray.direction[2]))
        if dist < maxDist - EPS:
            hit = True

    return hit, dist, normal

@ti.func
def HitLight(ray, maxDist):
    return HitSphere(ray.origin, ray.direction, Vec3f([0.0, 10.0, 0.0]), 2.0, maxDist)

@ti.func
def ReadVolume(pos):
    return Vec3f([1, 1, 1]), lerp(noise.Noise3D(pos) * 0.5 + 0.5, 0.0, 0.1)

@ti.func
def CalcVolumeRayDist(density):
    return ti.min(-ti.log(ti.random(ti.f32)) / density, 1e20)

@ti.func
def Trace(ray, maxDist):
    attenuation = Vec3f([1.0, 1.0, 1.0])
    debug = 0.0

    hitLight = False
    color = Zero3f
    absorbed = False

    for i in range(depth):
        if hitLight or absorbed: continue
        
        hitL, distL = HitLight(ray, maxDist)
        hitG, distG, N = HitGround(ray, maxDist)

        if hitG and ((not hitL) or distL >= distG):
            nextOrigin = ray.origin + distG * ray.direction
            nextDir = (N + RandomUnitVec3()).normalized()
            ray = TRay(origin=nextOrigin, direction=nextDir)
            attenuation *= Vec3f([1.0, 1.0, 1.0])
            if nextDir[2] <= EPS: absorbed = True

        elif hitL and ((not hitG) or distG >= distL):
            debug = 1.0
            nextOrigin = ray.origin + distL * ray.direction
            color = Vec3f([1.0, 0.8, 0.6]) * 10
            hitLight = True

        else:
            nextOrigin = ray.origin + maxDist * ray.direction
            nextDir = (Vec3f([0.0, 0.0, 1.0]) + RandomUnitVec3() * 0.8).normalized()
            albedo, density = ReadVolume(nextOrigin)
            attenuation *= albedo
            ray = TRay(origin=nextOrigin, direction=nextDir)
            maxDist = CalcVolumeRayDist(density)

    # debug = 1.0 if absorbed else 0.0
    return color * attenuation, debug
    # return debug * XUnit3f, debug
    # return ray.direction[1] * XUnit3f, debug

@ti.kernel
def Render():
    for u, v in sceneColorBuffer:
        color = Zero3f

        ray = CastRay(u + ti.random(ti.f32) - 0.5, v + ti.random(ti.f32) - 0.5)
        albedo, density = ReadVolume(cameraPos[None])
        c, debug = Trace(ray, CalcVolumeRayDist(density))
        # c, debug = Trace(ray, 1e20)
        color += c
        debugBuffer[u, v] = debug

        accum = accumulate[None]
        sceneColorBuffer[u, v] = (sceneColorBuffer[u, v] * (accum - 1) + color) / accum

@ti.kernel
def Clear():
    for u, v in sceneColorBuffer:
        sceneColorBuffer[u, v] = Zero3f

@ti.kernel
def Present():
    for i, j in sceneColorBuffer:
        windowImage[i, j] = sceneColorBuffer[i, j]

def PrintDebugBuffer():
    buf = debugBuffer.to_numpy()
    np.savetxt("./temp/out.csv", buf, delimiter=',', fmt='%.2f')

while window.running:
    elapsedTime[None] = float(time.time() - beginTime)
    
    updated = cameraControl.UpdateCamera(window, cameraTrans)
    
    fov[None] = 1.0
    cameraPos[None] = cameraTrans.mPos
    cameraDir[None] = cameraTrans.mDir
    up[None] = cameraTrans.mUp
    right[None] = cameraTrans.mRight

    if updated:
        Clear()
        accumulate[None] = 1
    else:
        accumulate[None] += 1

    Render()

    if window.is_pressed('p'):
        PrintDebugBuffer()
        break
        
    Present()
    window.get_canvas().set_image(windowImage)
    window.show()