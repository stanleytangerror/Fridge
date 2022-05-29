import taichi as ti

Vec2i = ti.types.vector(2, ti.i32)
Vec2f = ti.types.vector(2, ti.f32)
Vec3f = ti.types.vector(3, ti.f32)

EPS = 0.0001

@ti.func
def lerp(t, a, b): 
    return a + t * (b - a)

@ti.func
def saturate(v): 
    return ti.max(0.0, ti.min(1.0, v))
        
@ti.func
def AlmostEqual(a, b):
    return ti.abs(a - b) < EPS

@ti.func
def RandomUnitDisk():
    v = Vec2f([ti.random(ti.f32) * 2.0 - 1.0, ti.random(ti.f32) * 2.0 - 1.0])
    while v.norm_sqr() >= 1.0:
        v = Vec2f([ti.random(ti.f32) * 2.0 - 1.0, ti.random(ti.f32) * 2.0 - 1.0])
    return v

@ti.func
def RandomUnitVec3():
    phi = ti.random(ti.f32) * 2.0 * ti.math.pi
    z = ti.random(ti.f32) * 2.0 - 1.0
    r = ti.sqrt(1.0 - z ** 2)
    return Vec3f([r * ti.cos(phi), r * ti.sin(phi), z])

@ti.func
def HitSphere(rayOrigin, rayDir, sphereCenter, sphereRadius, maxDist):
    offset = rayOrigin - sphereCenter
    a = rayDir.norm_sqr()
    b = 2.0 * (offset[0] * rayDir[0] + offset[1] * rayDir[1] + offset[2] * rayDir[2])
    c = offset.norm_sqr() - sphereRadius ** 2
    k = b ** 2 - 4 * a * c

    dist = maxDist
    hit = False

    if k >= 0:
        t = (-b - ti.sqrt(k)) / 2 / a
        if t > 0: dist = min(dist, t)
        t = (-b + ti.sqrt(k)) / 2 / a
        if t > 0: dist = min(dist, t)

        if dist < maxDist - EPS:
            hit = True
    return hit, dist

@ti.func
def HitPlaneNormDist(rayOrigin, rayDir, planeNormal, planeD, maxDist):
    hit = False
    dist = maxDist

    NdotD = rayDir.dot(planeNormal)
    if ti.abs(NdotD) > EPS:
        NdotO = rayOrigin.dot(planeNormal)
        dist = ti.min(dist, -(planeD + NdotO) / NdotD)
        if dist > EPS:
            hit = True

    return hit, dist

@ti.func
def HitPlaneNormPoint(rayOrigin, rayDir, planeNormal, planePoint, maxDist):
    planeD = -planeNormal.dot(planePoint)
    return HitPlaneNormDist(rayOrigin, rayDir, planeNormal, planeD, maxDist)

@ti.func
def HitAABB(rayOrigin, rayDir, boxMin, boxMax, maxDist):
    hit = True

    tNear = maxDist
    tFar = -maxDist

    toMin = boxMin - rayOrigin
    toMax = boxMax - rayOrigin

    for i in ti.static(range(3)):
        if not hit: 
            continue
        elif abs(rayDir[i]) < EPS:
            if rayOrigin[i] < boxMin[i] or rayOrigin[i] > boxMax[i]:
                hit = False
        else:
            t1 = toMin[i] / rayDir[i]
            t2 = toMax[i] / rayDir[i]

            tNear = ti.min(tNear, ti.min(t1, t2))
            tFar = ti.max(tFar, ti.max(t1, t2))

    if tNear > tFar or tFar < 0:
        hit = False

    t = tFar
    goOut = True
    if tNear > 0:
        t = tNear
        goOut = False

    return hit, t, goOut

if __name__ == "__main__":
    import time

    ti.init(arch=ti.vulkan)

    WindowSize = (1920, 1080)
    windowImage = ti.Vector.field(3, float, shape=WindowSize)
    elapsedTime = ti.field(dtype=ti.f32, shape=())

    @ti.kernel
    def TestHitSphere():
        for i, j in windowImage:
            origin = Vec3f([i - WindowSize[0]/2, -1000, j - WindowSize[1]/2])
            dir = Vec3f([0, 1, 0])
            center = Vec3f([ti.cos(elapsedTime[None]), 0, ti.sin(elapsedTime[None])]) * 100.0
            hit, dist = HitSphere(origin, dir, center, 300.0, 1e20)
            if hit:
                norm = (origin + dist * dir - center).normalized()
                windowImage[i, j] = norm * 0.5 + 0.5

    window = ti.ui.Window("TestHit", WindowSize, vsync=True)
    beginTime = time.time()

    while window.running:
        windowImage.fill(0)
        elapsedTime[None] = float(time.time() - beginTime)
        TestHitSphere()
        window.get_canvas().set_image(windowImage)
        window.show()