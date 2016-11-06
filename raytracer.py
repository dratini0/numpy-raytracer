#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function, division, unicode_literals
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from math import *
import time

print("imports done")

PREFERRED_DTYPE = np.float64

Sphere = namedtuple("Sphere", ["centre", "radius", "material"])
Material = namedtuple("Material", ["ambientColor", "diffuseColor", "phongColor", "phongN"])
Light = namedtuple("Light", ["pos", "color"])
Camera = namedtuple("Camera", ["pos", "up", "lookAt", "hfov", "ratio"])
CastingResult = namedtuple("CastingResult", ["pos", "normal", "direction", "origin", "objectId", "zBuf"]) #candidate extensions: shaderID, UV
GlobalSettings = namedtuple("GlobalSettings", ["bgColor", "ambient"])

def vec3(x, y, z, dtype=PREFERRED_DTYPE):
    return np.array([x, y, z], dtype=dtype)

def vec4(x, y, z, w, dtype=PREFERRED_DTYPE):
    return np.array([x, y, z, w], dtype=dtype)

def vecabs(a):
    return np.linalg.norm(a, axis=-1).reshape(a.shape[:-1] + (1,))

def normalize(a):
    return a/vecabs(a)

def simpleDot(a, b, axis=-1):
    return np.sum(a * b, axis=axis)

pic_width = 640
pic_height = 480
camera = Camera(vec3(0, 0, 0), vec3(0, 1, 0), vec3(0, 0, 1), 60, 4/3)
ambient = vec3(0x55/0xff, 0x55/0xff, 0x55/0xff)
lights = [Light(vec3(1, 3, 2), vec3(0xb3/0xff, 0xdd/0xff, 0xff/0xff)*100)]
objects = [Sphere(vec3(0.55, -0.16, 3.5), 0.5, Material(vec3(0x00/0xff, 0x71/0xff, 0xbc/0xff),
                                                        vec3(0x00/0xff, 0x71/0xff, 0xbc/0xff)*.8,
                                                        vec3(1, 1, 1)*.8,
                                                        10)),
           Sphere(vec3(-0.55, 0, 5), 0.9, Material(vec3(0xff/0xff, 0x1d/0xff, 0x25/0xff),
                                                   vec3(0xff/0xff, 0x1d/0xff, 0x25/0xff)*.8,
                                                   vec3(1, 1, 1)*.8,
                                                   10)),
           ]
globalSettings = GlobalSettings(vec3(0, 0, 0), vec3(1/3, 1/3, 1/3))

def castSphere(wip, sphereId, sphere):
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
    wip.pos[mask] = s * wip.direction[mask] + wip.origin
    wip.normal[mask] = normalize(wip.pos[mask] - sphere.centre)
    wip.objectId[mask] = sphereId

def cast(direction, origin, objects):
    # I'm afraid to use empty
    pos = np.zeros_like(direction)
    normal = np.zeros_like(direction)
    objectId = np.full(direction.shape[:-1] + (1,), -1, np.int32)
    zBuf = np.full(direction.shape[:-1] + (1,), np.inf, direction.dtype)
    wip = CastingResult (pos, normal, direction, origin, objectId, zBuf)
    for index, obj in enumerate(objects):
        # branc by type late
        castSphere(wip, index, obj)
    return wip

def shade(castingResult, objects, lights, globalSettings):
    colorBuf = np.empty_like(castingResult.pos)
    mask = (castingResult.objectId != -1)[...,0]
    colorBuf[~mask] = globalSettings.bgColor
    # Here per shader stuff starts
    ambientColor = np.empty_like(colorBuf[mask])
    diffuseColor = np.empty_like(ambientColor)
    phongColor = np.empty_like(ambientColor)
    phongN = np.empty_like(ambientColor[...,1])
    for i, obj in enumerate(objects):
        thisObject = castingResult.objectId[mask][...,0] == i
        ambientColor[thisObject] = obj.material.ambientColor
        diffuseColor[thisObject] = obj.material.diffuseColor
        phongColor[thisObject] = obj.material.phongColor
        phongN[thisObject] = obj.material.phongN

    ambient = ambientColor * globalSettings.ambient
    colorBuf[mask] = ambient

    incidentDirection = normalize(-castingResult.direction[mask])
    normal = castingResult.normal[mask]
    reflectDirection = (2 * simpleDot(normal, incidentDirection)[...,np.newaxis]) * normal - incidentDirection

    for light in lights:
        toLight = light.pos - castingResult.pos[mask]
        lightDirection = normalize(toLight)
        lightIntensity = light.color / pi / 4 / simpleDot(toLight, toLight)[...,np.newaxis]
        colorBuf[mask] += lightIntensity * diffuseColor * np.maximum(0, simpleDot(lightDirection, normal))[...,np.newaxis]
        colorBuf[mask] += lightIntensity * phongColor * (np.maximum(0, simpleDot(lightDirection, reflectDirection)) ** phongN)[...,np.newaxis]
        

    return colorBuf

def makePic(color):
    return np.maximum(0, np.minimum(1, color))

camera_dir = normalize(camera.lookAt - camera.pos)
camera_right = -np.cross(camera_dir, normalize(camera.up)) # left handed, apparently
camera_up = -np.cross(camera_right, camera_dir) # also left hand
camera_right_scale = tan(camera.hfov / 360 * pi) * 2
camera_up_scale = -camera_right_scale / camera.ratio # increasing numbers go down
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
ray_origin = camera.pos

print(time.perf_counter())
castingResult = cast(ray_directions, ray_origin, objects)

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

color = shade(castingResult, objects, lights, globalSettings)
print(time.perf_counter())
plt.imshow(makePic(color))
plt.show()
