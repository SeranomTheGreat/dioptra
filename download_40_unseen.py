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

def download_targeted_pairs(env_name, difficulty, target_specs, out_dir):
    """
    target_specs: list of tuples (traj, preferred_index_fraction)
    e.g. [('P000', 0.25), ('P001', 0.50), ('P002', 0.75), ('P004', 0.35)]
    """
    print(f"\n=======================================================")
    print(f"  Downloading targeted {env_name} ({difficulty}) pairs")
    print(f"=======================================================")
    os.makedirs(out_dir, exist_ok=True)
    
    img_url = f'https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main/{env_name}/{difficulty}/image_left.zip'
    depth_url = f'https://huggingface.co/datasets/theairlabcmu/tartanair/resolve/main/{env_name}/{difficulty}/depth_left.zip'
    
    print(f"Connecting to {env_name}/{difficulty} image archive...")
    zf_img = zipfile.ZipFile(HttpRangeFile(img_url))
    print(f"Connecting to {env_name}/{difficulty} depth archive...")
    zf_depth = zipfile.ZipFile(HttpRangeFile(depth_url))
    
    # Map (traj, stem) -> zip path
    img_map = {}
    for f in zf_img.namelist():
        if f.endswith('.png'):
            parts = f.split('/')
            if len(parts) >= 5 and parts[-2] == 'image_left':
                traj = parts[-3]
                stem = parts[-1].replace('.png', '')
                img_map[(traj, stem)] = f
                
    depth_map = {}
    for f in zf_depth.namelist():
        if f.endswith('_depth.npy'):
            parts = f.split('/')
            if len(parts) >= 5 and parts[-2] == 'depth_left':
                traj = parts[-3]
                stem = parts[-1].replace('_depth.npy', '')
                depth_map[(traj, stem)] = f
                
    common_keys = set(img_map.keys()) & set(depth_map.keys())
    print(f"Found {len(common_keys)} common (traj, stem) pairs in archive.")
    
    # Group common keys by trajectory
    by_traj = {}
    for traj, stem in sorted(common_keys):
        by_traj.setdefault(traj, []).append(stem)
        
    for traj, frac in target_specs:
        if traj not in by_traj:
            avail = sorted(list(by_traj.keys()))
            if not avail:
                print(f"No available trajectories for {env_name}/{difficulty}!")
                continue
            traj = avail[0]
            
        stems = by_traj[traj]
        idx = int(len(stems) * frac)
        idx = min(max(0, idx), len(stems) - 1)
        stem = stems[idx]
        
        target_name = f"{traj}_{stem}"
        img_out = os.path.join(out_dir, f"{target_name}.png")
        depth_out = os.path.join(out_dir, f"{target_name}_depth.npy")
        
        if os.path.exists(img_out) and os.path.exists(depth_out):
            print(f"Already exists: {target_name}")
            continue
            
        print(f"Fetching pair [{traj}]: {target_name} ({idx}/{len(stems)})...")
        img_path_in_zip = img_map[(traj, stem)]
        depth_path_in_zip = depth_map[(traj, stem)]
        
        img_data = zf_img.read(img_path_in_zip)
        depth_data = zf_depth.read(depth_path_in_zip)
        
        with open(img_out, 'wb') as f:
            f.write(img_data)
        with open(depth_out, 'wb') as f:
            f.write(depth_data)
        print(f"  -> Saved {target_name} ({len(img_data)} B img, {len(depth_data)} B depth)")

if __name__ == '__main__':
    # 1. abandonedfactory (Daylight Easy) - Add 4 pairs from distinct trajectories
    download_targeted_pairs('abandonedfactory', 'Easy', [
        ('P000', 0.25), ('P001', 0.50), ('P002', 0.75), ('P004', 0.35)
    ], 'test_samples/abandonedfactory')
    
    # 2. abandonedfactory (Daylight Hard) - Add 4 pairs from distinct trajectories
    download_targeted_pairs('abandonedfactory', 'Hard', [
        ('P000', 0.30), ('P001', 0.60), ('P002', 0.40), ('P003', 0.70)
    ], 'test_samples/abandonedfactory_hard')
    
    # 3. abandonedfactory_night (Night Relit Easy) - Add 4 pairs from distinct trajectories
    download_targeted_pairs('abandonedfactory_night', 'Easy', [
        ('P001', 0.25), ('P002', 0.50), ('P003', 0.75), ('P004', 0.35)
    ], 'test_samples/abandonedfactory_night')
    
    # 4. abandonedfactory_night (Night Relit Hard) - Add 4 pairs from distinct trajectories
    download_targeted_pairs('abandonedfactory_night', 'Hard', [
        ('P000', 0.30), ('P001', 0.60), ('P002', 0.40), ('P003', 0.70)
    ], 'test_samples/abandonedfactory_night_hard')
    
    # 5. amusement (Outdoor Easy) - Add 4 pairs from distinct trajectories
    download_targeted_pairs('amusement', 'Easy', [
        ('P001', 0.25), ('P002', 0.50), ('P003', 0.75), ('P004', 0.35)
    ], 'test_samples/amusement')
    
    # 6. amusement (Outdoor Hard) - Add 4 pairs from distinct trajectories
    download_targeted_pairs('amusement', 'Hard', [
        ('P000', 0.30), ('P001', 0.60), ('P002', 0.40), ('P003', 0.70)
    ], 'test_samples/amusement_hard')
    
    print("\n=======================================================")
    print("  ALL 24 ADDITIONAL UNSEEN HELD-OUT DOWNLOADS COMPLETE!")
    print("=======================================================")
