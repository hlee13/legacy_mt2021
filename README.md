# misc

# tensorflow demo
https://www.tuicool.com/articles/vyiYJzu
https://github.com/tobegit3hub/tensorflow_template_application

class PointsCompression(object):
  def __init__(self):
	  self.table = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-'

  def parseToPoints(self, s):
    points = s.split(",")
    result = []
    for p in points:
        coord = p.split(" ")
        result.append([float(coord[1]), float(coord[0])])
    return result

  def evaluate(self, s, percision=100000):
    points = self.parseToPoints(s)
    lat, lon = 0, 0
    result = []
    for point in points:
      newLat = int(round(point[0] * percision))
      newLon = int(round(point[1] * percision))
      dx = newLat - lat
      dy = newLon - lon
      record = str([newLat, newLon])+' [ '+str(dx)+', '+str(dy)
      lat = newLat
      lon = newLon
      # 循环左移: *2, 负数转正数-1
      dy = (dy << 1) ^ (dy >> 31)
      dx = (dx << 1) ^ (dx >> 31)
      index = (dx + dy) * (dx + dy + 1) / 2 + dx
      record = record +'] [' + str(index) + ' ]'
      while index > 0:
        rem = index & 31
        index = (index - rem) / 32
        if index > 0:
            rem = rem + 32
        result.append(self.table[rem])
    return "".join(result)

const table = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-';
function decompressPoints(s) {
  const array = s.split('');
  const nums = [];
  let num = 0;
  let i = array.length - 1;
  for (; i >= 1; i -= 1) {
    const current = table.indexOf(array[i]);
    const previous = table.indexOf(array[i - 1]);
    if (current < 32) {
      num = current * 32;
    } else {
      num += current - 32;
      if (previous < 32) {
        nums.push(num);
      } else {
        num *= 32;
      }
    }
  }
  nums.push(num);
  const current = table.indexOf(array[i]);
  nums[nums.length - 1] = nums[nums.length - 1] + (current - 32);
  const coords = nums.map((e) => {
    let x = Math.sqrt(e * 2);
    x = Math.floor(x);
    let lat = e - x * (x + 1) / 2;
    while (lat < 0) {
      x -= 1;
      lat = e - x * (x + 1) / 2;
    }
    let lon = x - lat
    if (lat % 2 === 0) {
      lat /= 2;
    } else {
      lat = -((lat + 1) / 2);
    }
    if (lon % 2 === 0) {
      lon /= 2;
    } else {
      lon = -((lon + 1) / 2);
    }
    return [lat, lon];
  })
  for (let j = coords.length - 2; j >= 0; j -= 1) {
    const latLon = coords[j];
    latLon[0] += coords[j + 1][0];
    latLon[1] += coords[j + 1][1];
  }
  const result = coords.map((e) => {
    // return [e[1] / 100000.0, e[0] / 100000.0];
    return new window.AMap.LngLat(e[1] / 100000.0, e[0] / 100000.0);
  });
  return result;
}
