"""
@Author    : zhjm
@Time      : 2025/12/31 
@File      : demo.py
@Desc      : 
"""

# python - << 'EOF'
class A():
    def b(self):
        print(1)
class B(A):
    def c(self):
        print(2)
if __name__ == "__main__":
    print(B.__bases__)