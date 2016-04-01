# сейчас пока кусками читать особого смысла нет, т.к оказалось, что большую
# часть времени ест pymorphy, поэтому пока читаю все сразу, если надо будет,
# адаптировать достаточно легко под необработанные данные и задать размер
# кусков, которые он будет выдавать

def read_chunk(f, delimiter):
    buf = ""
    while True:
        while delimiter in buf:
            pos = buf.index(delimiter)
            yield buf[:pos]
            buf = buf[pos + len(delimiter):]
        chunk = f.read(2048)
        if not chunk:
            yield buf
            break
        buf += chunk