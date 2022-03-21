"""This module is heavily inspired by
https://github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/rendering.py
This refactor adds GPU support
"""
from typing import Callable, Tuple
import jax
import jax.image
import jax.numpy as jnp
from jax.numpy import ndarray as Array

BLACK = jnp.array([0, 0, 0], dtype=jnp.uint8)
RED = jnp.array([255, 0, 0], dtype=jnp.uint8)
GREEN = jnp.array([0, 255, 0], dtype=jnp.uint8)
BLUE = jnp.array([0, 0, 255], dtype=jnp.uint8)
PURPLE = jnp.array([112, 30, 195], dtype=jnp.uint8)
YELLOW = jnp.array([0, 255, 255], dtype=jnp.uint8)
ORANGE = jnp.array([255, 128, 0], dtype=jnp.uint8)
WHITE = jnp.array([255, 255, 255], dtype=jnp.uint8)
GREY_100 = jnp.array([100, 100, 100], dtype=jnp.uint8)
GREY_50 = jnp.array([50, 50, 50], dtype=jnp.uint8)
BACKGROUND = jnp.array([0, 0, 0], dtype=jnp.uint8)

MaskingFunction = Callable[[Array, Array], Array]
Point = Tuple[float, float]

# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32
FRAME_THICKNESS = 0.001


def get_grid() -> Array:
    """A computation grid for raster geometries.
    The space is reparameterised between [0, 1] with `TILE_PIXELS` number of tiles.
    """
    step = 1 / (TILE_PIXELS - 1)
    return jnp.mgrid[0 : 1 + step: step, 0 : 1 + step: step]


def asuint8(array: Array) -> Array:
    return jnp.asarray(array, dtype=jnp.uint8)


def downsample(img: Array, factor: int) -> Array:
    """
    Downsample an image along both dimensions by some factor.
    Notice that, while the original implementations uses mean interpolation,
    here we use nearest neighbour interpolation

    Args:
        img (Array): The image to resize in height and width
        factor (int): The factor by which the dimensions should be divided by

    Returns:
        (Array): The downsampled image
    """
    assert img.shape[0] % factor == 0
    assert img.shape[1] % factor == 0

    reshape = (img.shape[0] // factor, img.shape[1] // factor, *img.shape[2:])
    img = jax.image.resize(img, reshape, "bilinear")
    return asuint8(img)


def upsample(img: Array, factor: int) -> Array:
    """
    Upsample an image along both dimensions by some factor.
    Notice that, while the original implementations uses mean interpolation,
    here we use nearest neighbour interpolation

    Args:
        img (Array): The image to resize in height and width
        factor (int): The factor by which the dimensions should be multiplied by

    Returns:
        (Array): The upsampled image
    """
    if not isinstance(factor, int):
        raise ValueError(
            "`factor` must be an instance of type `int`, got {} instead".format(
                type(factor)
            )
        )
    reshape = (img.shape[0] * factor, img.shape[1] * factor, *img.shape[2:])
    img = jax.image.resize(img, reshape, "bilinear")
    return asuint8(img)


def apply(mask: Array, colour: Array, background: Array = BACKGROUND) -> Array:
    img = jnp.where(mask, colour, background)
    return asuint8(img)


def one_colour(colour: Array) -> Array:
    tile = jnp.ones((TILE_PIXELS, TILE_PIXELS, 3)) * colour
    return asuint8(tile)


def rectangle(
    a: Point, b: Point, colour: Array, background: Array = BACKGROUND
) -> Array:
    if a is None or b is None:
        raise ValueError("One of the rectangle points is null")

    x, y = get_grid()
    x_min, y_min = a
    x_max, y_max = b

    mask = jnp.logical_and(x >= x_min, x <= x_max)
    mask = jnp.logical_and(mask, y >= y_min)
    mask = jnp.logical_and(mask, y <= y_max)
    mask = jnp.stack([mask] * 3, axis=-1)

    tile = jnp.where(mask, colour, background)
    return asuint8(tile)


def circle(
    centre: Point, r: float, colour: Array, background: Array = BACKGROUND
) -> Array:
    if centre is None:
        raise ValueError("Centre is null")

    x, y = get_grid()
    c_x, c_y = centre

    mask = (((x - c_x) * (x - c_x) + (y - c_y) * (y - c_y)) <= r) * r
    mask = jnp.stack([mask] * 3, axis=-1)

    tile = jnp.where(mask, colour, background)
    return asuint8(tile)


def triangle(
    a: Point, b: Point, c: Point, colour: Array, background: Array = BACKGROUND
) -> Array:
    if a is None or b is None or c is None:
        raise ValueError("One of the triangle points is null")

    x, y = get_grid()
    a_x, a_y = a
    b_x, b_y = b
    c_x, c_y = c

    # get barycentric coordinates
    area = 0.5 * (-b_y * c_x + a_y * (-b_x + c_x) + a_x * (b_y - c_y) + b_x * c_y)
    sign = (area >= 0) * 2 - 1
    s = (a_y * c_x - a_x * c_y + (c_y - a_y) * x + (a_x - c_x) * y) * sign
    t = (a_x * b_y - a_x * b_x + (a_y - b_y) * x + (b_x - a_x) * y) * sign

    # inclusion test
    mask = jnp.logical_and(s >= 0, t >= 0)
    mask = jnp.logical_and(mask, s + t < 2 * area * sign)
    mask = jnp.stack([mask] * 3, axis=-1)

    tile = jnp.where(mask, colour, background)
    return asuint8(tile)


def floor(
    thickness: float = FRAME_THICKNESS,
    colour: Array = BLACK,
    background: Array = GREY_100,
) -> Array:
    t = thickness
    return rectangle((t, t), (1 - t, 1 - t), colour, background)


def block(colour: Array, size: float = 0.98, background: Array = BACKGROUND) -> Array:
    return floor(1 - size, colour=colour, background=floor())


def agent(colour: Array = RED, background: Array = BACKGROUND) -> Array:
    a = (0.12, 0.19)
    b = (0.12, 0.81)
    c = (0.87, 0.50)
    background = floor()
    return triangle(a, b, c, colour, background)


def key(colour: Array = BLUE, background: Array = BACKGROUND) -> Array:
    # trunk
    tile = rectangle((0.50, 0.31), (0.63, 0.88), colour)
    # teeth
    tile = rectangle((0.38, 0.59), (0.5, 0.66), colour, tile)
    tile = rectangle((0.38, 0.81), (0.5, 0.88), colour, tile)
    # ring
    tile = circle((0.56, -0.28), 0.19, colour, tile)
    tile = circle((0.56, -0.28), 0.064, BLACK, tile)
    return tile


def highlight(tile: Array, alpha: float = 0.3) -> Array:
    return asuint8(jnp.clip(tile + alpha * 256, 0, 256))
