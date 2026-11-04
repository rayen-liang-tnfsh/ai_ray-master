import sys


def solve(data):
    idx = 0
    n = int(data[idx]); idx += 1
    arr = [int(data[idx + i]) for i in range(n)]
    idx += n
    t = int(data[idx]); idx += 1

    quit_flag = False
    for _ in range(t):
        a = int(data[idx]); b = int(data[idx + 1]); idx += 2
        if quit_flag:
            continue
        if abs(a - b) > 8:
            quit_flag = True
            continue
        arr[a - 1], arr[b - 1] = arr[b - 1], arr[a - 1]

    head = "I QUIT!" if quit_flag else "SORTED!"
    return head + "\n" + " ".join(map(str, arr))


if __name__ == "__main__":
    data = sys.stdin.buffer.read().split()
    print(solve(data))
