from Math import *
from Camera import *
import numpy as np

ti.init(arch=ti.vulkan)

TRect = ti.types.struct(v0=Vec3f, v1=Vec3f, v2=Vec3f, v3=Vec3f, normal=Vec3f, emissive=Vec3f, albedo=Vec3f)
TRay = ti.types.struct(origin=Vec3f, direction=Vec3f)

def CreateRect(vertices, emissive, color):
    norm = (vertices[2] - vertices[1]).cross(vertices[1] - vertices[0]).normalized()
    return TRect(v0 = vertices[0], v1 = vertices[1], v2 = vertices[2], v3 = vertices[3], normal=norm, emissive=emissive, albedo=color)

White = Vec3f([1, 1, 1])
Black = Vec3f([0, 0, 0])
Zero3f = Vec3f([0.0, 0.0, 0.0])
One3f = Vec3f([1.0, 1.0, 1.0])
XUnit3f = Vec3f([1.0, 0.0, 0.0])
YUnit3f = Vec3f([0.0, 1.0, 0.0])
ZUnit3f = Vec3f([0.0, 0.0, 1.0])

rectList = []
rectList.append(CreateRect([
    Vec3f([552.8,   0.0,   0.0  ]),
    Vec3f([0.0,     0.0,   0.0  ]),
    Vec3f([0.0,     0.0, 559.2  ]),
    Vec3f([549.6,   0.0, 559.2  ]) ], 
    Black, White))
rectList.append(CreateRect([
    Vec3f([343.0,   548.8, 227.0   ]),
    Vec3f([343.0,   548.8, 332.0  ]),
    Vec3f([213.0,   548.8, 332.0  ]),
    Vec3f([213.0,   548.8, 227.0  ]) ], 
    Vec3f([8.0, 15.6, 18.4]), Black))

rects = TRect.field(shape=(len(rectList)))
for i in range(len(rectList)):
    rects[i] = rectList[i]


WindowSize = (1920, 1080)
window = ti.ui.Window("CornellBox", WindowSize, vsync=True)

cameraTrans = CameraTransform(pos=Vec3f([278, 273, -800]), dir=ZUnit3f, up=YUnit3f)
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

sceneColorBuffer = Vec3f.field(shape=WindowSize)
windowImage = ti.Vector.field(3, float, shape=WindowSize)
debugBuffer = ti.field(dtype=ti.f32, shape=WindowSize)

spp = 1
maxDepth = 100

@ti.func
def CastRay(u, v):
    offset = aperture * 0.5 * RandomUnitDisk()
    o = cameraPos[None] + offset[0] * right[None] + offset[1] * up[None]

    u = u + ti.random(ti.f32)
    v = v + ti.random(ti.f32)
    fu = 0.5 * fov[None] * (u * invResolution[0] - 0.5) * aspectRatio
    fv = 0.5 * fov[None] * (v * invResolution[1] - 0.5)
    disp = focusDist * Vec3f([ fu, fv, 1.0 ])
        
    d = (cameraPos[None] + disp[2] * cameraDir[None] + disp[0] * right[None] + disp[1] * up[None] - o).normalized()

    return TRay(origin=o, direction=d)

@ti.func
def HitScene(ray, maxDist):
    hit = False
    dist = maxDist
    norm = ZUnit3f
    emissive = Zero3f
    albedo = Zero3f

    for i in range(rects.shape[0]):
        rect = rects[i]
        h, d = HitPolygon(ray.origin, ray.direction, [rect.v0, rect.v1, rect.v2, rect.v3], maxDist)
        if h and (not hit or (hit and d < dist)):
            hit = True
            dist = d
            norm = rect.normal
            emissive = rect.emissive
            albedo = rect.albedo

    return hit, dist, norm, emissive, albedo

@ti.func
def Trace(ray, maxDist):
    brk = False
    att = One3f
    color = Zero3f

    for bounce in range(maxDepth):
        if brk: continue
        if ti.random() > 0.9:
            brk = True

        hit, dist, norm, emissive, albedo = HitScene(ray, maxDist)
        if hit:
            if emissive.max() > EPS:
                brk = True
                color = emissive
            else:
                att *= albedo
                nextOrigin = ray.origin + dist * ray.direction
                nextDir = (Reflect(norm, -ray.direction) + RandomUnitVec3()).normalized()
                ray = TRay(origin=nextOrigin, direction=nextDir)
        else:
            brk = True

    return att * color, 1

@ti.kernel
def Render():
    for u, v in sceneColorBuffer:
        color = Zero3f

        for i in range(spp):
            ray = CastRay(u + ti.random(ti.f32) - 0.5, v + ti.random(ti.f32) - 0.5)
            c, debug = Trace(ray, 1e20)
            color += c
            debugBuffer[u, v] = debug
        color = color / spp

        accum = accumulate[None]
        sceneColorBuffer[u, v] = (sceneColorBuffer[u, v] * (accum - 1) + color) / accum

@ti.kernel
def Present():
    for i, j in sceneColorBuffer:
        windowImage[i, j] = sceneColorBuffer[i, j]

def DebugColorBuffer():
    buf = debugBuffer.to_numpy()
    np.savetxt("temp/out.csv", buf, delimiter=',', fmt='%.2f')

while window.running:

    updated = cameraControl.UpdateCamera(window, cameraTrans)
    
    fov[None] = 1.0
    cameraPos[None] = cameraTrans.mPos
    cameraDir[None] = cameraTrans.mDir
    up[None] = cameraTrans.mUp
    right[None] = cameraTrans.mRight

    if updated:
        sceneColorBuffer.fill(0)
        accumulate[None] = 1
    else:
        accumulate[None] += 1

    Render()
       
    if window.is_pressed('p'):
        DebugColorBuffer()
        break

    Present()
    window.get_canvas().set_image(windowImage)
    window.show()
