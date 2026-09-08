import urllib.request, io, zipfile, os

class HttpRangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.pos = 0
        req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            self.length = int(resp.headers.get('Content-Length', 0))
    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET: self.pos = offset
        elif whence == io.SEEK_CUR: self.pos += offset
        elif whence == io.SEEK_END: self.pos = self.length + offset
        return self.pos
    def tell(self): return self.pos
    def read(self, size=-1):
        if size == -1 or self.pos + size > self.length: size = self.length - self.pos
        if size <= 0: return b''
        req = urllib.request.Request(self.url, headers={'Range': f'bytes={self.pos}-{self.pos+size-1}', 'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp: data = resp.read()
        self.pos += len(data)
        return data

def download_environment_samples(env_name, indices, out_dir):
    print(f"\n--- Downloading {env_name} samples ---")
    os.makedirs(out_dir, exist_ok=True)
    
    img_url = f'https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main/{env_name}/Easy/image_left.zip'
    depth_url = f'https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main/{env_name}/Easy/depth_left.zip'
    
    print(f"Connecting to {env_name} image archive...")
    zf_img = zipfile.ZipFile(HttpRangeFile(img_url))
    print(f"Connecting to {env_name} depth archive...")
    zf_depth = zipfile.ZipFile(HttpRangeFile(depth_url))
    
    img_files = {os.path.basename(f).replace('.png', ''): f for f in zf_img.namelist() if f.endswith('.png')}
    depth_files = {os.path.basename(f).replace('_depth.npy', ''): f for f in zf_depth.namelist() if f.endswith('_depth.npy')}
    
    common = sorted(list(set(img_files.keys()) & set(depth_files.keys())))
    print(f"Total matching pairs: {len(common)}")
    
    for idx in indices:
        if idx >= len(common):
            idx = len(common) - 1
        target = common[idx]
        img_out = os.path.join(out_dir, f"{target}.png")
        depth_out = os.path.join(out_dir, f"{target}_depth.npy")
        
        if os.path.exists(img_out) and os.path.exists(depth_out):
            print(f"Already exists: {target}")
            continue
            
        print(f"Fetching pair [{idx}/{len(common)}]: {target}...")
        img_data = zf_img.read(img_files[target])
        depth_data = zf_depth.read(depth_files[target])
        
        with open(img_out, 'wb') as f:
            f.write(img_data)
        with open(depth_out, 'wb') as f:
            f.write(depth_data)
        print(f"Saved {target} (image: {len(img_data)} B, depth: {len(depth_data)} B)")

if __name__ == '__main__':
    # 1. Amusement (Held-Out Test Environment)
    download_environment_samples('amusement', [50, 250, 600], 'test_samples/amusement')
    
    # 2. Abandoned Factory Night (Held-Out Test Environment - Night Lighting)
    download_environment_samples('abandonedfactory_night', [100, 400], 'test_samples/abandonedfactory_night')
    
    # 3. Hospital (Indoor Environment)
    download_environment_samples('hospital', [50, 300], 'test_samples/hospital')
    
    print("\nAll downloads complete!")
