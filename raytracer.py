#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function, division, unicode_literals
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from math import *
import time
import sys
import scipy.misc

print("imports done")

PREFERRED_DTYPE = np.float64

Sphere = namedtuple("Sphere", ["centre", "radius", "material"])
Plane = namedtuple("Plane", ["point", "normal", "material"])
Material = namedtuple("Material", ["ambientColor", "diffuseColor", "phongColor", "phongN", "reflect"])
Light = namedtuple("Light", ["pos", "color"])
Camera = namedtuple("Camera", ["pos", "up", "lookAt", "hfov", "ratio"])
GlobalSettings = namedtuple("GlobalSettings", ["bgColor", "ambient"])
Scene = namedtuple("Scene", ["camera", "lights", "objects", "globalSettings"])
CastingResult = namedtuple("CastingResult", ["pos", "normal", "direction", "origin", "objectId", "zBuf"]) #candidate extensions: shaderID, UV

def vec3(x, y, z, dtype=PREFERRED_DTYPE):
    return np.array([x, y, z], dtype=dtype)

def vec4(x, y, z, w, dtype=PREFERRED_DTYPE):
    return np.array([x, y, z, w], dtype=dtype)

def vecabs(a):
    return np.linalg.norm(a, axis=-1)[..., np.newaxis]

def normalize(a):
    return a/vecabs(a)

def simpleDot(a, b, axis=-1):
    return np.sum(a * b, axis=axis)

def makeSimpleMaterial1(color):
    return Material(color, color*.8, vec3(1, 1, 1)*.8, 10, 0)

def makeSimpleMaterial2Sphere(color):
    return Material(color, color*.8, vec3(1, 1, 1)*.8, 10, .3)

def makeSimpleMaterial2Plane(color):
    return Material(color, color*.9, vec3(1, 1, 1)*.0, 0, .1)

def colorFromHex(color):
    if color[0] == '#': color = color[1:]
    asBytes = bytes.fromhex(color)
    return vec3(asBytes[0]/0xff, asBytes[1]/0xff, asBytes[2]/0xff)

pic_width = 640
pic_height = 480
depth = 2
reflectEpsilon = 0.0001
scenes = [
Scene(
    camera = Camera(vec3(0, 0, 0), vec3(0, 1, 0), vec3(0, 0, 1), 45, 4/3),
    lights = [Light(vec3(1, 3, 2), colorFromHex("#B3DDFF")*100)],
    objects = [Sphere(vec3(0.55, -0.16, 3.5), 0.5, makeSimpleMaterial1(colorFromHex("#0071BC"))),
               Sphere(vec3(-0.55, 0, 5), 0.9, makeSimpleMaterial1(colorFromHex("#FF1D25"))),
               ],
    globalSettings = GlobalSettings(vec3(.1, .1, .1), vec3(1/3, 1/3, 1/3))
),
Scene(
    camera = Camera(vec3(0, 0, 0), vec3(0, 1, 0), vec3(0, 0, 1), 45, 4/3),
    lights = [Light(vec3(2, 2, 4.5), colorFromHex("#FFB0B2")*100),
              Light(vec3(-2, 2.5, 1), colorFromHex("#FFF5CC")*200),
              ],
    objects = [Sphere(vec3(-0.95, -0.21884, 3.63261), 0.35, makeSimpleMaterial2Sphere(colorFromHex("#FF1D25"))),
               Sphere(vec3(-0.4, 0.5, 4.33013), 0.7, makeSimpleMaterial2Sphere(colorFromHex("#0071BC"))),
               Sphere(vec3(0.72734, -0.35322, 3.19986), 0.45, makeSimpleMaterial2Sphere(colorFromHex("#3AA010"))),
               Plane(vec3(.0, -0.10622, 4.68013), vec3(0, 4.2239089012146, -2.180126190185547), makeSimpleMaterial2Plane(colorFromHex("#222222"))),
               ],
    globalSettings = GlobalSettings(vec3(.1, .1, .1), vec3(1/3, 1/3, 1/3))
)
]

def castSphere(sphere, wip, sphereId):
    a = (wip.direction * wip.direction).sum(axis=-1, keepdims=True)
    b = ((wip.origin - sphere.centre) * wip.direction).sum(axis=-1, keepdims=True)
    b = b * 2
    c_ = wip.origin - sphere.centre
    c_ = (c_ * c_).sum(axis=-1, keepdims=True) - sphere.radius * sphere.radius
    c = np.empty_like(a)
    c[...] = c_

    D = b * b - 4 * a * c
    mask = (D[..., 0] >= 0)
    plt.plot()
    s = (- b[mask] - np.sqrt(D[mask])) / (2 * a[mask])
    mask2 = s[..., 0] <= 0
    s[mask2] = (- b[mask][mask2] + np.sqrt(D[mask][mask2])) / (2 * a[mask][mask2])
    del mask2
    mask3 = s[..., 0] > 0
    mask[mask] = mask3
    s = s[mask3]
    del mask3
    mask4 = (s < wip.zBuf[mask])[..., 0]
    mask[mask] = mask4
    s = s[mask4]
    del mask4
    wip.zBuf[mask] = s
    originMask = wip.origin if wip.origin.ndim == 1 else wip.origin[mask]
    wip.pos[mask] = s * wip.direction[mask] + originMask
    wip.normal[mask] = normalize(wip.pos[mask] - sphere.centre)
    wip.objectId[mask] = sphereId
Sphere.cast = castSphere

def castPlane(self, wip, objId):
    s = simpleDot(self.point - wip.origin, self.normal) / simpleDot(wip.direction, self.normal)
    mask = (s > 0) & (s < wip.zBuf[..., 0])
    s = s[..., np.newaxis][mask]
    wip.zBuf[mask] = s
    originMask = wip.origin if wip.origin.ndim == 1 else wip.origin[mask]
    wip.pos[mask] = s * wip.direction[mask] + originMask
    wip.normal[mask] = normalize(self.normal)
    wip.objectId[mask] = objId

Plane.cast = castPlane

def cast(direction, origin, objects):
    # I'm afraid to use empty
    pos = np.zeros_like(direction)
    normal = np.zeros_like(direction)
    objectId = np.full(direction.shape[:-1] + (1,), -1, np.int32)
    zBuf = np.full(direction.shape[:-1] + (1,), np.inf, direction.dtype)
    wip = CastingResult (pos, normal, direction, origin, objectId, zBuf)
    for index, obj in enumerate(objects):
        obj.cast(wip, index)
    return wip

def shade(castingResult, objects, lights, globalSettings, depth):
    colorBuf = np.empty_like(castingResult.pos)
    mask = (castingResult.objectId != -1)[...,0]
    colorBuf[~mask] = globalSettings.bgColor
    # Here per shader stuff starts
    ambientColor = np.empty_like(colorBuf[mask])
    diffuseColor = np.empty_like(ambientColor)
    phongColor = np.empty_like(ambientColor)
    phongN = np.empty_like(ambientColor[...,1])
    reflect = np.empty_like(phongN)
    for i, obj in enumerate(objects):
        thisObject = castingResult.objectId[mask][...,0] == i
        ambientColor[thisObject] = obj.material.ambientColor
        diffuseColor[thisObject] = obj.material.diffuseColor
        phongColor[thisObject] = obj.material.phongColor
        phongN[thisObject] = obj.material.phongN
        reflect[thisObject] = obj.material.reflect

    ambient = ambientColor * globalSettings.ambient
    colorBuf[mask] = ambient

    incidentDirection = normalize(-castingResult.direction[mask])
    normal = castingResult.normal[mask]
    reflectDirection = (2 * simpleDot(normal, incidentDirection)[...,np.newaxis]) * normal - incidentDirection

    for light in lights:
        toLight = light.pos - castingResult.pos[mask]
        shadowTrace = cast(-toLight, light.pos, objects)
        lightHits = castingResult.objectId[mask][..., 0] == shadowTrace.objectId[..., 0]
        mask2 = mask.copy()
        mask2[mask2] = lightHits
        toLight = toLight[lightHits]
        lightDirection = normalize(toLight)
        lightIntensity = light.color / pi / 4 / simpleDot(toLight, toLight)[...,np.newaxis]
        colorBuf[mask2] += lightIntensity * diffuseColor[lightHits] * np.maximum(0, simpleDot(lightDirection, normal[lightHits]))[...,np.newaxis]
        colorBuf[mask2] += lightIntensity * phongColor[lightHits] * (np.maximum(0, simpleDot(lightDirection, reflectDirection[lightHits])) ** phongN[lightHits])[...,np.newaxis]

    if depth > 0:
        reflectionCast = cast(reflectDirection, castingResult.pos[mask] + reflectDirection * reflectEpsilon, objects)
        reflectionColor = shade(reflectionCast, objects, lights, globalSettings, depth - 1)
        colorBuf[mask] = colorBuf[mask] * (1 - reflect)[..., np.newaxis] + reflectionColor * reflect[..., np.newaxis]

    return colorBuf

def normalizePic(color):
    return np.maximum(0, np.minimum(1, color))

sceneNumber = int(sys.argv[1]) - 1
scene = scenes[sceneNumber]

camera_dir = normalize(scene.camera.lookAt - scene.camera.pos)
camera_right = -np.cross(camera_dir, normalize(scene.camera.up)) # left handed, apparently
camera_up = -np.cross(camera_right, camera_dir) # also left hand
camera_right_scale = tan(scene.camera.hfov / 360 * pi) * 2
camera_up_scale = -camera_right_scale / scene.camera.ratio # increasing numbers go down
x_ray_range = np.arange(0.5, pic_width, dtype=PREFERRED_DTYPE)
x_ray_range.shape = (1, pic_width, 1)
x_ray_range /= float(pic_width)
x_ray_range -= 0.5
x_ray_range = x_ray_range * (camera_right_scale * camera_right)
y_ray_range = np.arange(0.5, pic_height, dtype=PREFERRED_DTYPE)
y_ray_range.shape = (pic_height, 1, 1)
y_ray_range /= float(pic_height)
y_ray_range -= 0.5
y_ray_range = y_ray_range * (camera_up_scale * camera_up)
ray_directions = (camera_dir + y_ray_range) + x_ray_range
ray_origin = scene.camera.pos

print(time.perf_counter())
castingResult = cast(ray_directions, ray_origin, scene.objects)

#plt.imshow(castingResult.objectId[...,0])
#plt.figure()
#plt.imshow(castingResult.zBuf[...,0])
#plt.figure()
#plt.imshow((castingResult.zBuf*vecabs(castingResult.direction))[...,0])
#plt.figure()
#plt.imshow(castingResult.normal)
#plt.figure()
#plt.imshow(castingResult.pos)
#plt.show()

color = shade(castingResult, scene.objects, scene.lights, scene.globalSettings, depth)
print(time.perf_counter())

color = normalizePic(color)
if len(sys.argv) >= 3:
    scipy.misc.toimage(color, cmin=0.0, cmax=1.0).save(sys.argv[2])
else:
    plt.imshow(color)
    plt.show()

