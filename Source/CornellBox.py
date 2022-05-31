from Math import *
from Camera import *
import numpy as np

@ti.data_oriented
class CornellBox:
    TRect = ti.types.struct(v0=Vec3f, v1=Vec3f, v2=Vec3f, v3=Vec3f, normal=Vec3f, mat=ti.i32)
    TRectWithHole = ti.types.struct(v0=Vec3f, v1=Vec3f, v2=Vec3f, v3=Vec3f, vh0=Vec3f, vh1=Vec3f, vh2=Vec3f, vh3=Vec3f, normal=Vec3f, mat=ti.i32)

    Light = 0
    WhiteSurface = 1
    GreenSurface = 2
    RedSurface = 3

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
    def CreateRect(vertices, mat):
        norm = (vertices[2] - vertices[1]).cross(vertices[1] - vertices[0]).normalized()
        return CornellBox.TRect(   v0 = vertices[0], v1 = vertices[1], v2 = vertices[2], v3 = vertices[3], 
                        normal=norm, mat=mat)

    @staticmethod
    def CreateRectWithHole(vertices, holeVertices, mat):
        norm = (vertices[2] - vertices[1]).cross(vertices[1] - vertices[0]).normalized()
        return CornellBox.TRectWithHole(   
                        v0 = vertices[0], v1 = vertices[1], v2 = vertices[2], v3 = vertices[3], 
                        vh0 = holeVertices[0], vh1 = holeVertices[1], vh2 = holeVertices[2], vh3 = holeVertices[3], 
                        normal=norm, mat=mat)

    @staticmethod
    def CreateRects(rectVertices, mat):
        result = []
        for vertices in rectVertices:
            norm = (vertices[2] - vertices[1]).cross(vertices[1] - vertices[0]).normalized()
            result.append(CornellBox.TRect(v0 = vertices[0], v1 = vertices[1], v2 = vertices[2], v3 = vertices[3], normal=norm, mat=mat))
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
            CornellBox.WhiteSurface)

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
        mat = 0

        for i in range(self.rects.shape[0]):
            rect = self.rects[i]
            h, d = HitPolygon(rayOrigin, rayDir, [rect.v0, rect.v1, rect.v2, rect.v3], maxDist)
            if h and (not hit or (hit and d < dist)):
                hit = True
                dist = d
                norm = rect.normal
                mat = rect.mat

        for i in range(self.rectsWithHole.shape[0]):
            rect = self.rectsWithHole[i]
            h, d = self.HitPolygonWithHole(rayOrigin, rayDir, [rect.v0, rect.v1, rect.v2, rect.v3], 
                                                                [rect.vh0, rect.vh1, rect.vh2, rect.vh3], maxDist)
            if h and (not hit or (hit and d < dist)):
                hit = True
                dist = d
                norm = rect.normal
                mat = rect.mat

        if hit:
            if norm.dot(-rayDir) < 0.0:
                norm *= -1.0

        return hit, dist, norm, mat

if __name__ == "__main__":

    TRay = ti.types.struct(origin=Vec3f, direction=Vec3f)

    ti.init(arch=ti.vulkan)
    cornellBox = CornellBox()

    White = Vec3f([1, 1, 1])
    Red = Vec3f([1, 0, 0])
    Green = Vec3f([0, 1, 0])
    Zero3f = Vec3f([0.0, 0.0, 0.0])
    One3f = Vec3f([1.0, 1.0, 1.0])
    XUnit3f = Vec3f([1.0, 0.0, 0.0])
    YUnit3f = Vec3f([0.0, 1.0, 0.0])
    ZUnit3f = Vec3f([0.0, 0.0, 1.0])


    WindowSize = (1920, 1080)
    window = ti.ui.Window("CornellBox", WindowSize, vsync=False)

    pos, dir, up = cornellBox.GetCameraTransform()
    cameraTrans = CameraTransform(pos=pos, dir=dir, up=up)
    lens = CameraLens(90.0, WindowSize, 10, 0.0)
    cameraControl = CameraControl()

    cameraData = CameraDataConstBuffer()
    accumulate = ti.field(dtype=ti.f32, shape=())
    sceneColorBuffer = Vec3f.field(shape=WindowSize)
    windowImage = ti.Vector.field(3, float, shape=WindowSize)
    debugBuffer = ti.field(dtype=ti.f32, shape=WindowSize)

    spp = 4
    maxDepth = 100


    @ti.func
    def CastRay(u, v):
        offset = cameraData.aperture[None] * 0.5 * RandomUnitDisk()
        o = cameraData.cameraPos[None] + offset[0] * cameraData.right[None] + offset[1] * cameraData.up[None]

        u = u + ti.random(ti.f32)
        v = v + ti.random(ti.f32)
        fu = 0.5 * cameraData.fov[None] * (u * cameraData.invResolution[None][0] - 0.5) * cameraData.aspectRatio[None]
        fv = 0.5 * cameraData.fov[None] * (v * cameraData.invResolution[None][1] - 0.5)
        disp = cameraData.focusDist[None] * Vec3f([ fu, fv, 1.0 ])
            
        d = (cameraData.cameraPos[None] + disp[2] * cameraData.cameraDir[None] + disp[0] * cameraData.right[None] + disp[1] * cameraData.up[None] - o).normalized()

        return TRay(origin=o, direction=d)

    @ti.func
    def RandomUnitVec3OnHemisphere(n):
        v = RandomUnitVec3()
        if n.dot(v) < 0.0:
            v *= -1.0
        return v

    @ti.func
    def Trace(ray, maxDist):
        brk = False
        att = One3f
        color = Zero3f

        for bounce in range(maxDepth):
            p = att.max()
            if ti.random() > p:
                brk = True

            if brk: continue

            hit, dist, norm, mat = cornellBox.HitScene(ray.origin, ray.direction, maxDist)
            if hit:
                if mat == CornellBox.Light:
                    brk = True
                    color = One3f * 50.0
                else:
                    albedo =    Red if mat == CornellBox.RedSurface else \
                                Green if mat == CornellBox.GreenSurface else \
                                White
                    fBrdf = 1.0 / (2.0 * ti.math.pi)
                    pMC = 1.0 / (2.0 * ti.math.pi) # mc integral pdf
                    Li = RandomUnitVec3OnHemisphere(norm)
                    att *= albedo * fBrdf * norm.dot(Li) / pMC

                    fRR = 1.0 / p # russian rulette factor
                    att *= fRR
                    
                    nextOrigin = ray.origin + dist * ray.direction
                    nextDir = Li
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
