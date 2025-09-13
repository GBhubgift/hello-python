import turtle as t

# ===== 可调参数 =====
scale = 1.0              # 整体缩放：0.8 更小，1.2 更大
radius = int(90 * scale) # 每个环半径
pen_w = int(10 * scale)  # 线条粗细
dx = int(220 * scale)    # 水平间距（左中右）
dy = int(110 * scale)    # 垂直间距（上排到下排）

# ===== 颜色（接近官方配色）=====
t.colormode(255)
COLORS = {
    "blue":   (0, 133, 199),
    "yellow": (244, 195, 0),
    "black":  (0, 0, 0),
    "green":  (0, 159, 61),
    "red":    (223, 0, 36),
}

# ===== 画一个以 (cx, cy) 为圆心的环 =====
def draw_ring(cx, cy, rgb):
    t.pencolor(rgb)
    t.penup()
    t.goto(cx, cy - radius)  # 到圆底部的起点
    t.setheading(0)
    t.pendown()
    t.circle(radius)

# ===== 画布设置 =====
t.setup(width=900, height=600)
t.title("奥运五环 - turtle")
t.bgcolor("white")
t.hideturtle()
t.speed(0)        # 0 最快
t.pensize(pen_w)

# 上排：蓝、黑、红
draw_ring(-dx,     0, COLORS["blue"])
draw_ring(0,       0, COLORS["black"])
draw_ring(dx,      0, COLORS["red"])
# 下排：黄、绿
draw_ring(-dx//2, -dy, COLORS["yellow"])
draw_ring(dx//2,  -dy, COLORS["green"])

t.done()
