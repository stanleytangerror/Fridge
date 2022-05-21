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

depth = 100

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

smokeField = ti.field(dtype=ti.f32, shape=(256, 256, 64))
smokeBound = Vec3f.field(shape=(2))

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
def SampleField(pos):
    fieldShape = Vec3f(smokeField.shape)
    p = pos - ti.floor(pos / fieldShape) * fieldShape
    x = ti.floor(p)
    w = p - x
    return lerp(w[2],   lerp(w[1],  lerp(w[0], smokeField[x + Vec3f([0, 0, 0])], smokeField[x + Vec3f([1, 0, 0])]) ,
                                    lerp(w[0], smokeField[x + Vec3f([0, 1, 0])], smokeField[x + Vec3f([1, 1, 0])]) ),
                        lerp(w[1],  lerp(w[0], smokeField[x + Vec3f([0, 0, 1])], smokeField[x + Vec3f([1, 0, 1])]) ,
                                    lerp(w[0], smokeField[x + Vec3f([0, 1, 1])], smokeField[x + Vec3f([1, 1, 1])]) ))

@ti.func
def ReadVolume(pos):
    h = SampleField(pos * 20.0)
    # h = noise.Noise3D(pos)
    return Vec3f([0.9, 0.8, 0.7]), lerp(saturate(h * 0.5 + 0.5), 0.0, 0.5)

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
        
        # light
        hitL, distL = HitSphere(ray.origin, ray.direction, Vec3f([0.0, 10.0, 0.0]), 1.0, maxDist) 
        # ground
        hitG, distG = HitPlaneNormPoint(ray.origin, ray.direction, Vec3f([0.0, -1.0, 1.0]).normalized(), Vec3f([0.0, 20.0, 0.0]), maxDist)
        N = ZUnit3f

        if hitG and ((not hitL) or distL >= distG):
            nextOrigin = ray.origin + distG * ray.direction
            nextDir = (N + RandomUnitVec3()).normalized()
            ray = TRay(origin=nextOrigin, direction=nextDir)
            attenuation *= Vec3f([0.9, 0.9, 0.9])
            if nextDir[2] <= EPS: absorbed = True

        elif hitL and ((not hitG) or distG >= distL):
            debug = 1.0
            nextOrigin = ray.origin + distL * ray.direction
            color = Vec3f([1.0, 0.8, 0.6]) * 100.0
            hitLight = True

        else:
            nextOrigin = ray.origin + maxDist * ray.direction
            nextDir = (Vec3f([0.0, 0.0, 1.0]) + RandomUnitVec3() * 1.0).normalized()
            albedo, density = ReadVolume(nextOrigin)
            attenuation *= albedo
            ray = TRay(origin=nextOrigin, direction=nextDir)
            maxDist = CalcVolumeRayDist(density)

        prob = attenuation.max()
        if ti.random(ti.f32) > prob:
            absorbed = True
        else:
            attenuation /= prob
        

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
        color += c
        debugBuffer[u, v] = debug

        accum = accumulate[None]
        sceneColorBuffer[u, v] = (sceneColorBuffer[u, v] * (accum - 1) + color) / accum


        # sceneColorBuffer[u, v] = One3f * SampleField(Vec3f([u, v, 5]) * 0.02)

        # sceneColorBuffer[u, v] = One3f * smokeField[u % smokeField.shape[0], v % smokeField.shape[1], 5]

@ti.kernel
def Clear():
    for u, v in sceneColorBuffer:
        sceneColorBuffer[u, v] = Zero3f

@ti.kernel
def Present():
    for i, j in sceneColorBuffer:
        windowImage[i, j] = sceneColorBuffer[i, j]

@ti.kernel
def GenerateFbm3D():
    for i, j, k in smokeField:
        pos = Vec3f([i, j, k])
        h = 0.0
        freq = 0.1
        amp = 1.0
        for l in range(1, 20):
            m = pow(2.0, float(l))
            h += noise.Noise3D(pos * freq * m) * amp / m
        smokeField[i, j, k] = h

def PrintDebugBuffer():
    buf = debugBuffer.to_numpy()
    np.savetxt("./temp/out.csv", buf, delimiter=',', fmt='%.2f')


GenerateFbm3D()
smokeBound[0] = Vec3f([-20.0, -20.0, -20.0])
smokeBound[1] = Vec3f([ 20.0,  20.0,  20.0])

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