import taichi as ti

Vec2i = ti.types.vector(2, ti.i32)
Vec2f = ti.types.vector(2, ti.f32)
Vec3f = ti.types.vector(3, ti.f32)
Vec4f = ti.types.vector(4, ti.f32)
Mat33f = ti.types.matrix(3, 3, ti.f32)

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
def AlmostZero(a):
    return AlmostEqual(a, 0.0)

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
def Reflect(N, L):
    assert N.dot(L) > 0.0
    O = N.dot(L) * 2.0 * N - L
    assert AlmostEqual(O.norm(), 1.0)
    return O

@ti.func
def Refract(N, L, EtaL_Over_EtaO):
    assert N.dot(L) > 0.0
    cosL = N.dot(L)
    sinL = ti.sqrt(1.0 - cosL ** 2)
    sinO = EtaL_Over_EtaO * sinL
    assert sinO <= 1.0
    cosO = ti.sqrt(1.0 - sinO ** 2)
    
    oHori = -EtaL_Over_EtaO * (L - cosL * N)
    oVert = -N * cosO
    O = oHori + oVert
    assert AlmostEqual(O.norm(), 1.0)
    return O

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

@ti.func
def HitTriangle(rayOrigin, rayDir, v0, v1, v2, maxDist):
    # Möller Trumbore algo http://www.graphics.cornell.edu/pubs/1997/MT97.pdf
    
    hit = False
    dist = maxDist

    E1 = v1 - v0
    E2 = v2 - v0
    T = rayOrigin - v0

    P = rayDir.cross(E2)
    Q = T.cross(E1)
    PdotE1 = P.dot(E1)
    
    if abs(PdotE1) > EPS:
        invPdotE1 = 1.0 / PdotE1
        t = Q.dot(E2) * invPdotE1
        b1 = P.dot(T) * invPdotE1
        b2 = Q.dot(rayDir) * invPdotE1

        if EPS < t and t <= maxDist and \
            0.0 <= b1 and 0.0 <= b2 and b1 + b2 <= 1.0:
            hit = True
            dist = t

    return hit, dist

@ti.func
def HitPolygon(rayOrigin, rayDir, vertices, maxDist):
    assert len(vertices) >= 3

    hit = False
    dist = maxDist

    for i in ti.static(range(len(vertices) - 2)):
        h, d = HitTriangle(rayOrigin, rayDir, vertices[0], vertices[i+1], vertices[i+2], maxDist)
        if h:
            hit = True
            dist = ti.min(dist, d)
                
    return hit, dist

@ti.func
def HitQuadric3d(rayOrigin, rayDir, Q, P, R, maxDist):
    # https://en.wikipedia.org/wiki/Quadric
    # Quadric def: x^t * Q * x + P^T * x + R = 0, Q Mat33, P Vec3, R Scalar
    # quadratic equation form of t (x = O + tD) like https://www.cs.uaf.edu/2012/spring/cs481/section/0/lecture/01_26_ray_intersections.html

    hit = False
    dist = maxDist

    O = rayOrigin
    D = rayDir
    a = D.dot(Q@D)
    b = D.dot(Q@O) + O.dot(Q@D) + P.dot(D)
    c = P.dot(O) + O.dot(Q@O) + R

    if AlmostZero(a):
        if AlmostZero(b):
            if AlmostZero(c):
                dist = EPS
        else:
            t = -c / b
            if t > EPS: dist = min(t, dist)
    else:
        ss = b * b-4.0 * a * c
        if ss >= 0:
            sss = ti.sqrt(ss)
            t = (-b - sss) / 2.0 / a
            if t > EPS: dist = min(t, dist)
            t = (-b + sss) / 2.0 / a
            if t > EPS: dist = min(t, dist)
    
    if 0.0 < dist and dist < maxDist - EPS:
        hit = True
    
    return hit, dist

@ti.func
def CalcQuadric3dNormal(x, Q, P, R):
    # Quadric def: x^t * Q * x + P^T * x + R = 0, Q Mat33, P Vec3, R Scalar
    # calc derivative from https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf
    return ((Q + Q.transpose()) @ x + P).normalized()

if __name__ == "__main__":
    import time
    from Camera import *

    ti.init(arch=ti.vulkan)

    WindowSize = (1920, 1080)
    windowImage = ti.Vector.field(3, float, shape=WindowSize)
    elapsedTime = ti.field(dtype=ti.f32, shape=())

    window = ti.ui.Window("TestHit", WindowSize, vsync=True)
    beginTime = time.time()

    cameraTrans = CameraTransform(pos=Vec3f([0, -100, 0]), dir=Vec3f([0, 1, 0]), up=Vec3f([0, 0, 1]))
    lens = CameraLens(90.0, WindowSize, 10, 0.0)
    cameraControl = CameraControl()
    cameraData = CameraDataConstBuffer()

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

    @ti.kernel
    def TestHitTriangle():
        for i, j in windowImage:
            origin = Vec3f([i - WindowSize[0]/2, -1000, j - WindowSize[1]/2])
            dir = Vec3f([0, 1, 0])

            t1 = elapsedTime[None] * 0.7
            t2 = elapsedTime[None] * 1.1
            t3 = elapsedTime[None] * 0.3
            vertices = [
                Vec3f([ti.cos(t1), 0, ti.sin(t1)]) * 200.0,
                Vec3f([-ti.sin(t2), ti.cos(t2), 0]) * 200.0,
                Vec3f([0, ti.cos(t3), ti.sin(t3)]) * 200.0,
            ]
            hit, dist = HitTriangle(origin, dir, vertices, 1e20)

            if hit:
                norm = (vertices[1] - vertices[0]).cross(vertices[2] - vertices[0])
                if norm.norm() > EPS:
                    windowImage[i, j] = norm.normalized() * 0.5 + 0.5

    @ti.kernel
    def TestHitPolygon():
        for i, j in windowImage:
            origin = Vec3f([i - WindowSize[0]/2, -1000, j - WindowSize[1]/2])
            dir = Vec3f([0, 1, 0])

            t1 = elapsedTime[None] * 0.7
            t2 = elapsedTime[None] * 1.1
            t3 = elapsedTime[None] * 0.3

            v0 = Vec3f([ti.cos(t1), 0, ti.sin(t1)]) * 200.0
            e1 = Vec3f([-ti.sin(t2), ti.cos(t2), 0]) * 200.0 - v0
            e2 = Vec3f([0, ti.cos(t3), ti.sin(t3)]) * 200.0 - v0
            
            vertices = [
                v0,
                v0 + 1.0 * e1 + 0.0 * e2,
                v0 + 1.5 * e1 + 0.8 * e2,
                v0 + 1.0 * e1 + 1.0 * e2,
                v0 + 0.9 * e1 + 1.3 * e2,
                v0 + 0.0 * e1 + 1.0 * e2,
            ]
            hit, dist = HitPolygon(origin, dir, vertices, 1e20)

            if hit:
                norm = (vertices[1] - vertices[0]).cross(vertices[2] - vertices[0])
                if norm.norm() > EPS:
                    windowImage[i, j] = norm.normalized() * 0.5 + 0.5

    @ti.kernel
    def TestHitQuadric3d():
        for i, j in windowImage:
            origin, dir = CastRayForPixel(i, j, cameraData)

            t = ti.sin(elapsedTime[None]) + 1.1

            Q = Mat33f([
                [1, 0, 0], 
                [0, 1, 0], 
                [0, 0, 0], 
                ])
            P = Vec3f([0, 0, -1])
            R = 0

            hit, dist = HitQuadric3d(origin, dir, Q, P, R, 1e20)
            if hit:
                p = origin + dist * dir
                norm = CalcQuadric3dNormal(p, Q, P, R)
                windowImage[i, j] = norm * 0.5 + 0.5

            center = Vec3f([10, 0, 0])
            hit, dist = HitSphere(origin, dir, center, 5.0, 1e20)
            if hit:
                p = origin + dist * dir
                windowImage[i, j] = (p - center).normalized() * 0.5 + 0.5

    while window.running:
        elapsedTime[None] = float(time.time() - beginTime)
        cameraControl.UpdateCamera(window, cameraTrans, speed=5.0)
        cameraData.FillData(lens, cameraTrans)

        windowImage.fill(0)
        TestHitQuadric3d()
        window.get_canvas().set_image(windowImage)
        window.show()