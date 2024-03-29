from Math import *
from Camera import *

@ti.data_oriented
class CornellBox:
    TRect = ti.types.struct(v0=Vec3f, v1=Vec3f, v2=Vec3f, v3=Vec3f, normal=Vec3f, id=ti.i32)
    TRectWithHole = ti.types.struct(v0=Vec3f, v1=Vec3f, v2=Vec3f, v3=Vec3f, vh0=Vec3f, vh1=Vec3f, vh2=Vec3f, vh3=Vec3f, normal=Vec3f, id=ti.i32)

    Light = 0
    WhiteSurface = 1
    GreenSurface = 2
    RedSurface = 3
    ChromeSurface = 4

    @staticmethod
    @ti.func
    def HitPolygonWithHole(rayOrigin, rayDir, vertices, holeVertices, maxDist):
        assert len(vertices) >= 3

        hit = False
        dist = maxDist

        h, d = HitPolygon(rayOrigin, rayDir, vertices, maxDist)
        if h:
            hh, dh = HitPolygon(rayOrigin, rayDir, holeVertices, maxDist)
            if not hh:
                hit = True
                dist = d

        return hit, dist
    
    @staticmethod
    def CreateRect(vertices, id):
        norm = (vertices[2] - vertices[1]).cross(vertices[1] - vertices[0]).normalized()
        return CornellBox.TRect(   v0 = vertices[0], v1 = vertices[1], v2 = vertices[2], v3 = vertices[3], 
                        normal=norm, id=id)

    @staticmethod
    def CreateRectWithHole(vertices, holeVertices, id):
        norm = (vertices[2] - vertices[1]).cross(vertices[1] - vertices[0]).normalized()
        return CornellBox.TRectWithHole(   
                        v0 = vertices[0], v1 = vertices[1], v2 = vertices[2], v3 = vertices[3], 
                        vh0 = holeVertices[0], vh1 = holeVertices[1], vh2 = holeVertices[2], vh3 = holeVertices[3], 
                        normal=norm, id=id)

    @staticmethod
    def CreateRects(rectVertices, id):
        result = []
        for vertices in rectVertices:
            norm = (vertices[2] - vertices[1]).cross(vertices[1] - vertices[0]).normalized()
            result.append(CornellBox.TRect(v0 = vertices[0], v1 = vertices[1], v2 = vertices[2], v3 = vertices[3], normal=norm, id=id))
        return result

    def __init__(self):
        rectList = []
        # cornell box data http://www.graphics.cornell.edu/online/box/data.html
        rectList.append(self.CreateRect([ # floor
            Vec3f([ 552.8,   0.0,   0.0  ]),
            Vec3f([ 0.0,     0.0,   0.0  ]),
            Vec3f([ 0.0,     0.0, 559.2  ]),
            Vec3f([ 549.6,   0.0, 559.2  ]) ], 
            CornellBox.WhiteSurface))
        rectList.append(self.CreateRect([ # back wall
            Vec3f([ 549.6,   0.0, 559.2  ]),
            Vec3f([   0.0,   0.0, 559.2  ]),
            Vec3f([   0.0, 548.8, 559.2  ]),
            Vec3f([ 556.0, 548.8, 559.2  ]) ], 
            CornellBox.WhiteSurface))
        rectList.append(self.CreateRect([ # left wall
            Vec3f([ 552.8,   0.0,   0.0  ]),
            Vec3f([ 549.6,   0.0, 559.2  ]),
            Vec3f([ 556.0, 548.8, 559.2  ]),
            Vec3f([ 556.0, 548.8,   0.0  ]) ], 
            CornellBox.RedSurface))
        rectList.append(self.CreateRect([ # right wall
            Vec3f([ 0.0,   0.0, 559.2  ]),
            Vec3f([ 0.0,   0.0,   0.0  ]),
            Vec3f([ 0.0, 548.8,   0.0  ]),
            Vec3f([ 0.0, 548.8, 559.2  ]) ], 
            CornellBox.GreenSurface))
        rectList.append(self.CreateRect([ # light
            Vec3f([ 343.0,   548.8, 227.0  ]),
            Vec3f([ 343.0,   548.8, 332.0  ]),
            Vec3f([ 213.0,   548.8, 332.0  ]),
            Vec3f([ 213.0,   548.8, 227.0  ]) ], 
            CornellBox.Light))

        rectList += self.CreateRects([ # short block
                [
                    Vec3f([ 130.0, 165.0,  65.0 ]),
                    Vec3f([  82.0, 165.0, 225.0 ]),
                    Vec3f([ 240.0, 165.0, 272.0 ]),
                    Vec3f([ 290.0, 165.0, 114.0 ]) 
                ],
                [
                    Vec3f([ 290.0,   0.0, 114.0 ]),
                    Vec3f([ 290.0, 165.0, 114.0 ]),
                    Vec3f([ 240.0, 165.0, 272.0 ]),
                    Vec3f([ 240.0,   0.0, 272.0 ]) 
                ],
                [
                    Vec3f([ 130.0,   0.0,  65.0 ]),
                    Vec3f([ 130.0, 165.0,  65.0 ]),
                    Vec3f([ 290.0, 165.0, 114.0 ]),
                    Vec3f([ 290.0,   0.0, 114.0 ]) 
                ],
                [
                    Vec3f([  82.0,   0.0, 225.0 ]),
                    Vec3f([  82.0, 165.0, 225.0 ]),
                    Vec3f([ 130.0, 165.0,  65.0 ]),
                    Vec3f([ 130.0,   0.0,  65.0 ]) 
                ],
                [
                    Vec3f([ 240.0,   0.0, 272.0 ]),
                    Vec3f([ 240.0, 165.0, 272.0 ]),
                    Vec3f([  82.0, 165.0, 225.0 ]),
                    Vec3f([  82.0,   0.0, 225.0 ]) 
                ],
            ], 
            CornellBox.WhiteSurface)

        rectList += self.CreateRects([ # long block
                [
                    Vec3f([ 423.0, 330.0, 247.0 ]),
                    Vec3f([ 265.0, 330.0, 296.0 ]),
                    Vec3f([ 314.0, 330.0, 456.0 ]),
                    Vec3f([ 472.0, 330.0, 406.0 ]) 
                ],
                [
                    Vec3f([ 423.0,   0.0, 247.0 ]),
                    Vec3f([ 423.0, 330.0, 247.0 ]),
                    Vec3f([ 472.0, 330.0, 406.0 ]),
                    Vec3f([ 472.0,   0.0, 406.0 ]) 
                ],
                [
                    Vec3f([ 472.0,   0.0, 406.0 ]),
                    Vec3f([ 472.0, 330.0, 406.0 ]),
                    Vec3f([ 314.0, 330.0, 456.0 ]),
                    Vec3f([ 314.0,   0.0, 456.0 ]) 
                ],
                [
                    Vec3f([ 314.0,   0.0, 456.0 ]),
                    Vec3f([ 314.0, 330.0, 456.0 ]),
                    Vec3f([ 265.0, 330.0, 296.0 ]),
                    Vec3f([ 265.0,   0.0, 296.0 ]) 
                ],
                [
                    Vec3f([ 265.0,   0.0, 296.0 ]),
                    Vec3f([ 265.0, 330.0, 296.0 ]),
                    Vec3f([ 423.0, 330.0, 247.0 ]),
                    Vec3f([ 423.0,   0.0, 247.0 ]) 
                ],
            ], 
            CornellBox.ChromeSurface)

        rectWithHoleList = []
        rectWithHoleList.append(self.CreateRectWithHole( # ceiling
                [
                    Vec3f([ 556.0, 548.8, 0.0   ]),
                    Vec3f([ 556.0, 548.8, 559.2 ]),
                    Vec3f([   0.0, 548.8, 559.2 ]),
                    Vec3f([   0.0, 548.8,   0.0 ]) 
                ],
                [
                    Vec3f([ 343.0, 548.8, 227.0 ]),
                    Vec3f([ 343.0, 548.8, 332.0 ]),
                    Vec3f([ 213.0, 548.8, 332.0 ]),
                    Vec3f([ 213.0, 548.8, 227.0 ]) 
                ],
                CornellBox.WhiteSurface))

        self.rects = CornellBox.TRect.field(shape=(len(rectList)))
        for i in range(len(rectList)):
            self.rects[i] = rectList[i]

        self.rectsWithHole = CornellBox.TRectWithHole.field(shape=(len(rectWithHoleList)))
        for i in range(len(rectWithHoleList)):
            self.rectsWithHole[i] = rectWithHoleList[i]

    def GetCameraTransform(self):
        return Vec3f([278, 273, -800]), Vec3f([0, 0, 1]), Vec3f([0, 1, 0])  # pos, dir, up

    @ti.func
    def HitScene(self, rayOrigin, rayDir, maxDist):
        hit = False
        dist = maxDist
        norm = ZUnit3f
        id = 0

        for i in range(self.rects.shape[0]):
            rect = self.rects[i]
            h, d = HitPolygon(rayOrigin, rayDir, [rect.v0, rect.v1, rect.v2, rect.v3], maxDist)
            if h and (not hit or (hit and d < dist)):
                hit = True
                dist = d
                norm = rect.normal
                id = rect.id

        for i in range(self.rectsWithHole.shape[0]):
            rect = self.rectsWithHole[i]
            h, d = self.HitPolygonWithHole(rayOrigin, rayDir, [rect.v0, rect.v1, rect.v2, rect.v3], 
                                                                [rect.vh0, rect.vh1, rect.vh2, rect.vh3], maxDist)
            if h and (not hit or (hit and d < dist)):
                hit = True
                dist = d
                norm = rect.normal
                id = rect.id

        if hit:
            if norm.dot(-rayDir) < 0.0:
                norm *= -1.0

        return hit, dist, norm, id

if __name__ == "__main__":

    import numpy as np
    import OpenEXR
    from numpy import float16

    ti.init(arch=ti.vulkan)

    exrFile = OpenEXR.InputFile(r"D:\Download\box_exr\500nm.exr")
    print(exrFile.header())
    dw = exrFile.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    channelData = exrFile.channel('Y')
    baseImgContent = np.fromstring(channelData, dtype=np.float16)
    baseImgContent2d = np.fliplr(np.reshape(np.array(baseImgContent), (sz[1], sz[0])).transpose())
    baseImg = ti.field(dtype=ti.f32, shape=sz)
    baseImg.from_numpy(baseImgContent2d)
    
    TRay = ti.types.struct(origin=Vec3f, direction=Vec3f)

    cornellBox = CornellBox()

    White = Vec3f([1, 1, 1])
    Red = Vec3f([1, 0, 0])
    Green = Vec3f([0, 1, 0])
    Zero3f = Vec3f([0.0, 0.0, 0.0])
    One3f = Vec3f([1.0, 1.0, 1.0])
    XUnit3f = Vec3f([1.0, 0.0, 0.0])
    YUnit3f = Vec3f([0.0, 1.0, 0.0])
    ZUnit3f = Vec3f([0.0, 0.0, 1.0])


    WindowSize = (sz[0] * 2, sz[1])
    window = ti.ui.Window("CornellBox", WindowSize, vsync=False)

    pos, dir, up = cornellBox.GetCameraTransform()
    cameraTrans = CameraTransform(pos=pos, dir=dir, up=up)
    lens = CameraLens(90.0, sz, 10, 0.0)
    cameraControl = CameraControl()

    cameraData = CameraDataConstBuffer()
    accumulate = ti.field(dtype=ti.f32, shape=())
    sceneColorBuffer = Vec3f.field(shape=WindowSize)
    windowImage = ti.Vector.field(3, float, shape=WindowSize)
    debugBuffer = ti.field(dtype=ti.f32, shape=WindowSize)

    spp = 4.0
    maxDepth = 100

    @ti.func
    def RandomUnitVec3OnHemisphere(n):
        v = RandomUnitVec3()
        if n.dot(v) < 0.0:
            v *= -1.0
        return v

    @ti.func
    def RandomPointOnRect(rect):
        e1 = rect.v1 - rect.v0
        e3 = rect.v3 - rect.v0
        return rect.v0 + ti.random() * e1 + ti.random() * e3, e1.norm() * e3.norm()

    @ti.func
    def Trace(ray, maxDist):
        brk = False
        att = One3f
        color = Zero3f

        for bounce in range(maxDepth):
            if brk: continue

            hit, dist, norm, id = cornellBox.HitScene(ray.origin, ray.direction, maxDist)
            if hit:
                # shade hit point
                hitPoint = ray.origin + dist * ray.direction

                if id == CornellBox.Light: 
                    # hit light    
                    brk = True
                    color += att * 8.0
                else: 
                    # hit normal surface
                    p = att.max()
                    if ti.random() > p:
                        brk = True
                        continue
                        
                    # direct lighting contrib
                    for i in range(cornellBox.rects.shape[0]):
                        rect = cornellBox.rects[i]
                        if rect.id == CornellBox.Light:
                            pLight, A = RandomPointOnRect(rect)
                            hitToLight = pLight - hitPoint
                            distToLight = hitToLight.norm()
                            dirToLight = hitToLight / distToLight
                            
                            if dirToLight.dot(norm) > EPS:
                                hs, ds, ns, ids = cornellBox.HitScene(hitPoint, dirToLight, maxDist)
                                if hs and AlmostEqual(ds, distToLight):
                                    if id == CornellBox.ChromeSurface:
                                        Li = Reflect(norm, -ray.direction)
                                        if AlmostEqual(dirToLight[0], Li[0]) and AlmostEqual(dirToLight[1], Li[1]) and AlmostEqual(dirToLight[2], Li[2]):
                                            fBrdf = 1.0 # ???
                                            pMC = 1.0 / A
                                            L = 8.0 * fBrdf * norm.dot(dirToLight) / (distToLight ** 2) / pMC
                                            color += att * L
                                    else:
                                        albedo =    0.058 if id == CornellBox.RedSurface else \
                                                    0.285 if id == CornellBox.GreenSurface else \
                                                    0.747
                                        fBrdf = albedo / (2.0 * ti.math.pi) # lambertian
                                        pMC = 1.0 / A
                                        L = 8.0 * fBrdf * norm.dot(dirToLight) / (distToLight ** 2) / pMC
                                        color += att * L

                    # indirect lighting contrib
                    if id == CornellBox.ChromeSurface:
                        fBrdf = 1.0 # ???
                        Li = Reflect(norm, -ray.direction)
                        att *= fBrdf * norm.dot(Li)

                        fRR = 1.0 / p # russian rulette factor
                        att *= fRR

                        ray = TRay(origin=hitPoint, direction=Li)
                    else:
                        albedo =    0.058 if id == CornellBox.RedSurface else \
                                    0.285 if id == CornellBox.GreenSurface else \
                                    0.747
                        fBrdf = albedo / (2.0 * ti.math.pi) # lambertian
                        pMC = 1.0 / (2.0 * ti.math.pi) # mc integral pdf
                        Li = RandomUnitVec3OnHemisphere(norm)
                        att *= fBrdf * norm.dot(Li) / pMC

                        fRR = 1.0 / p # russian rulette factor
                        att *= fRR

                        ray = TRay(origin=hitPoint, direction=Li)
            else:
                brk = True

        return color

    @ti.kernel
    def Render():
        for u, v in sceneColorBuffer:
            color = Zero3f

            for i in range(spp):
                rayOrigin, rayDir = CastRayForPixel(u + ti.random(ti.f32) - 0.5, v + ti.random(ti.f32) - 0.5, cameraData)
                c = Trace(TRay(origin=rayOrigin, direction=rayDir), 1e20)
                color += c * One3f
            color = color / spp

            accum = accumulate[None]
            sceneColorBuffer[u, v] = (sceneColorBuffer[u, v] * (accum - 1) + color) / accum

    @ti.kernel
    def Present():
        for i, j in sceneColorBuffer:
            windowImage[i, j] = sceneColorBuffer[i, j]

    @ti.kernel
    def Present():
        for i, j in sceneColorBuffer:
            if i < sz[0]:
                windowImage[i, j] = sceneColorBuffer[i, j]
            else:
                windowImage[i, j] = baseImg[i - sz[0], j] * One3f
            windowImage[i, j] *= 2.0

    def DebugColorBuffer():
        buf = debugBuffer.to_numpy()
        np.savetxt("temp/out.csv", buf, delimiter=',', fmt='%.2f')

    while window.running:

        updated = cameraControl.UpdateCamera(window, cameraTrans, 50.0)
        
        cameraData.FillData(lens, cameraTrans)

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
