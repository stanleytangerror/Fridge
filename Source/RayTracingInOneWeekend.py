import imp
import taichi as ti
from taichi.math import *
import numpy as np
import math
import random
from Math import *
from Camera import *

# ---------------------------------------------------------------------

TSphere = ti.types.struct(center=Vec3f, radius=ti.f32, materialId=ti.i32, inverseNormal=ti.i32)
TRay = ti.types.struct(origin=Vec3f, direction=Vec3f)
TMaterial = ti.types.struct(type=ti.i32, emissive=Vec3f, albedo=Vec3f, roughness=ti.f32, refractionIndex=ti.f32, density=ti.f32)
TVolumeContext = ti.types.struct(inside=ti.i32, materialId=ti.i32)

Zero3f = Vec3f([0.0, 0.0, 0.0])
One3f = Vec3f([1.0, 1.0, 1.0])
XUnit3f = Vec3f([1.0, 0.0, 0.0])
YUnit3f = Vec3f([0.0, 1.0, 0.0])
ZUnit3f = Vec3f([0.0, 0.0, 1.0])

def RandVec3f():
    return Vec3f([random.random(), random.random(), random.random()])


# ---------------------------------------------------------------------
ti.init(arch=ti.vulkan)
# ti.init(arch=ti.cpu, debug=True, cpu_max_num_threads=1, advanced_optimization=False)
random.seed()

# WindowSize = (128, 72)
WindowSize = (1920, 1080)
RayMaxDistance = 1e20

Emissive = 0
Lambertian = 1
Metal = 2
Dielectric = 3
Volume = 4

def CreateLambertial(albedo, roughness):        return TMaterial(type=Lambertian, emissive=Zero3f, albedo=albedo, roughness=roughness, refractionIndex=1.0, density=1.0)
def CreateDielectric(albedo, refractionIndex):  return TMaterial(type=Dielectric, emissive=Zero3f, albedo=albedo, roughness=0.0, refractionIndex=refractionIndex, density=1.0)
def CreateMetal(albedo, roughness):             return TMaterial(type=Metal, emissive=Zero3f, albedo=albedo, roughness=roughness, refractionIndex=1.0, density=1.0)
def CreateVolume(albedo, density):              return TMaterial(type=Volume, emissive=Zero3f, albedo=albedo, roughness=0.0, refractionIndex=1.0, density=density)
def CreateEmissive(emissive):                   return TMaterial(type=Emissive, emissive=emissive, albedo=Zero3f, roughness=0.0, refractionIndex=1.0, density=1.0)

@ti.func
def ReadLambertial(mat):        assert mat.type == Lambertian;   return mat.albedo, mat.roughness
@ti.func
def ReadDielectric(mat):        assert mat.type == Dielectric;   return mat.albedo, mat.refractionIndex
@ti.func
def ReadMetal(mat):             assert mat.type == Metal;        return mat.albedo, mat.roughness
@ti.func
def ReadVolume(mat):            assert mat.type == Volume;       return mat.albedo, mat.density
@ti.func
def ReadEmissive(mat):          assert mat.type == Emissive;     return mat.emissive

# ---------------------------------------------------------------------
window = ti.ui.Window("RayTracingInOneWeekend", WindowSize, vsync=True)
cameraTrans = CameraTransform(-YUnit3f * 10.0)
lens = CameraLens(90.0, WindowSize, 10, 0.0)
cameraControl = CameraControl()

materialList = []
materialList.append(CreateLambertial(albedo=One3f, roughness=0.5))
materialList.append(CreateDielectric(albedo=One3f, refractionIndex=1.1))
materialList.append(CreateMetal(albedo=One3f, roughness=0.5))
materialList.append(CreateMetal(albedo=One3f, roughness=0.0))
materialList.append(CreateVolume(albedo=Vec3f([1.0, 0.8, 0.6]), density=1.5))
materialList.append(CreateEmissive(emissive=One3f * 10.0))
for i in range(5):
    materialList.append(CreateMetal(albedo=RandVec3f(), roughness=0.5))
    materialList.append(CreateMetal(albedo=RandVec3f(), roughness=0.0))
for i in range(5):
    materialList.append(CreateLambertial(albedo=RandVec3f(), roughness=1.0))
    materialList.append(CreateLambertial(albedo=RandVec3f(), roughness=0.0))
for i in range(5):
    materialList.append(CreateDielectric(albedo=One3f, refractionIndex=(1.0 + random.random())))

materials = TMaterial.field(shape=(len(materialList)))
for i in range(len(materialList)):
    materials[i] = materialList[i]

sphereList = []
sphereList.append(TSphere(center=Vec3f([30.0, 0.0, 20.0]), radius=10.0, materialId=5, inverseNormal=00))
sphereList.append(TSphere(center=Vec3f([0.0, 0.0, -1000.0]), radius=1000.0, materialId=0, inverseNormal=00))
sphereList.append(TSphere(center=Vec3f([-3.0, 0.0, 1.0]), radius=1, materialId=1, inverseNormal=00))
sphereList.append(TSphere(center=Vec3f([-3.0, 0.0, 1.0]), radius=0.9, materialId=1, inverseNormal=10))
for i in range(5):
    sphereList.append(TSphere(center=Vec3f([i * 3, 0.0, 1.0]), radius=1, materialId=i, inverseNormal=00))
for i in range(40):
    matId = random.randint(0, len(materialList))
    c = Vec3f([random.random() * 20.0 - 10.0, random.random() * 20.0 - 10.0, 0.3])
    sphereList.append(TSphere(center=c, radius=0.3, materialId=matId, inverseNormal=00))

spheres = TSphere.field(shape=(len(sphereList)))
for i in range(len(sphereList)):
    spheres[i] = sphereList[i]

spp = 1.0
depth = 100

cameraData = CameraDataConstBuffer()

cameraVolumeContext = TVolumeContext.field(shape=())
accumulate = ti.field(dtype=ti.f32, shape=())

sceneColorBuffer = Vec3f.field(shape=WindowSize)
windowImage = ti.Vector.field(3, float, shape=WindowSize)
debugBuffer = ti.field(dtype=ti.f32, shape=WindowSize)

# ---------------------------------------------------------------------

@ti.func
def HitSphereSurface(ray, sphere, maxDist):
    offset = ray.origin - sphere.center
    a = ray.direction.norm_sqr()
    b = 2.0 * (offset[0] * ray.direction[0] + offset[1] * ray.direction[1] + offset[2] * ray.direction[2])
    c = offset.norm_sqr() - sphere.radius ** 2
    k = b ** 2 - 4 * a * c

    dist = maxDist
    hit = False
    n = ZUnit3f
    matId = -1
    inside = False

    if k >= 0:
        t = (-b - ti.sqrt(k)) / 2 / a
        if t > 0: dist = min(dist, t)
        t = (-b + ti.sqrt(k)) / 2 / a
        if t > 0: dist = min(dist, t)

        if dist < maxDist - EPS:
            hit = True
            p = ray.origin + dist * ray.direction
            n = (p - sphere.center).normalized()
            if sphere.inverseNormal != 0: n *= -1.0
            matId = sphere.materialId
            assert AlmostEqual((p - sphere.center).norm(), sphere.radius)
            inside = n.dot(-ray.direction) < 0.0
    return hit, dist, n, inside, matId

@ti.func
def HitSceneSurface(ray, maxDist):
    hit = False
    dist = maxDist
    norm = ZUnit3f
    inside = False
    matId = -1

    for s in range(spheres.shape[0]):
        h, d, n, i, m = HitSphereSurface(ray, spheres[s], maxDist)
        if h and (not hit or (hit and d < dist)):
            hit = h
            dist = d
            norm = n
            inside = i
            matId = m
            assert matId >= 0
    return hit, dist, norm, inside, matId

@ti.func
def Trace(ray, volumeContext):
    attenuation = Vec3f([1.0, 1.0, 1.0])
    debug = 0

    hitLight = False
    color = Zero3f
    absorbed = False

    for i in range(depth):
        if hitLight or absorbed: continue
        
        if volumeContext.inside == 0:
            hit, dist, norm, inside, matId = HitSceneSurface(ray, RayMaxDistance)
            if hit:
                assert matId >= 0
                mat = materials[matId]
                hitPos = ray.origin + dist * ray.direction

                N = norm if not inside else -norm
                L = -ray.direction

                nextDir = Zero3f

                if mat.type == Lambertian:
                    if inside: absorbed = True
                    else:
                        nextDir = (N + RandomUnitVec3()).normalized()
                        albedo, roughness = ReadLambertial(mat)
                        attenuation *= albedo
                elif mat.type == Metal:
                    if inside: absorbed = True
                    else: 
                        albedo, roughness = ReadLambertial(mat)
                        attenuation *= albedo
                        nextDir = Zero3f
                        retryTime = 3
                        while nextDir.dot(norm) <= EPS and retryTime > 0:
                            nextDir = (Reflect(N, L) + RandomUnitVec3() * (roughness ** 2)).normalized()
                            retryTime -= 1
                        if nextDir.dot(norm) <= EPS: absorbed = True
                elif mat.type == Dielectric:
                    albedo, refrIdx = ReadDielectric(mat)
                    attenuation *= albedo
                    refrIndexLOverO = refrIdx if inside else (1.0 / refrIdx)
                    if refrIndexLOverO * ti.sqrt(1.0 - N.dot(L) ** 2) > 1.0:
                        nextDir = Reflect(N, L)
                    else:
                        nextDir = Refract(N, L, refrIndexLOverO)
                elif mat.type == Emissive:
                    hitLight = True
                    color = ReadEmissive(mat)
                elif mat.type == Volume:
                    # enter volume, do nothing
                    volumeContext = TVolumeContext(inside=1, materialId=matId)
                    nextDir = ray.direction 
                else:
                    absorbed = True
                    assert False

                if not absorbed:
                    ray = TRay(origin=hitPos + nextDir * EPS, direction=nextDir)
            else:
                absorbed = True
        else:
            assert volumeContext.materialId >= 0
            mat = materials[volumeContext.materialId]
            assert mat.type == Volume

            albedo, density = ReadVolume(mat)
            rayDist = ti.min(-ti.log(ti.random(ti.f32)) / density, RayMaxDistance)
            hit, dist, norm, inside, matId = HitSceneSurface(ray, rayDist)
            if hit:
                # escape from volume, do nothing
                hitPos = ray.origin + dist * ray.direction
                nextDir = ray.direction
                ray = TRay(origin=hitPos + nextDir * EPS, direction=nextDir)
                volumeContext = TVolumeContext(inside=0, materialId=-1)
            else:
                hitPos = ray.origin + ray.direction * rayDist
                nextDir = RandomUnitVec3()
                # nextDir = ray.direction
                ray = TRay(origin=hitPos, direction=nextDir)
                attenuation *= albedo

    debug = 1.0 if absorbed else 0.0
    return color * attenuation, debug

@ti.kernel
def Clear():
    for u, v in sceneColorBuffer:
        sceneColorBuffer[u, v] = Zero3f

@ti.kernel
def Render():
    for u, v in sceneColorBuffer:
        color = Zero3f

        for i in range(spp):
            rayOrigin, rayDir = CastRayForPixel(u + ti.random(ti.f32) - 0.5, v + ti.random(ti.f32) - 0.5, cameraData)
            ray = TRay(origin=rayOrigin, direction=rayDir)
            volCtx = cameraVolumeContext[None]
            c, debug = Trace(ray, volCtx)
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
    
    cameraData.FillData(lens, cameraTrans)

    cameraVolumeContext[None] = TVolumeContext(inside=0.0, materialId=-1)

    if updated:
        Clear()
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