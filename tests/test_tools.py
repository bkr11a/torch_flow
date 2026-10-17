import sys,unittest,tempfile,struct,zipfile,warnings
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from paired_bootstrap import bootstrap,paired_arrays
from export_benchmark import write_flow,verify_flow,pack,records

class Tests(unittest.TestCase):
    def test_constant_paired_effect(self):
        a=np.arange(40,dtype=float); b=a+2; groups=np.repeat(np.arange(10),4)
        r,_=bootstrap(a,b,groups,2000,8)
        self.assertEqual(r['delta'],-2); self.assertAlmostEqual(r['ci_low'],-2); self.assertAlmostEqual(r['ci_high'],-2)
    def test_pair_weighting_not_equal_scene_means(self):
        a=np.array([0.,0.,0.,8.]); b=np.zeros(4)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore'); r,loo=bootstrap(a,b,np.array(['a','a','a','b']),2000,2)
        self.assertEqual(r['A_mean'],2); self.assertEqual(r['n_clusters'],2)
        self.assertEqual({v['delta'] for v in loo},{0.,8.})
    def test_identity_pairing_and_validation(self):
        def doc(rows): return {'dataset':'x','evaluation':{},'rows':rows}
        rows=[{'pair_id':str(i),'cluster_id':str(i),'support_sha256':'s','metrics':{'epe_all':float(i)}} for i in range(4)]
        a,b,g=paired_arrays(doc(rows),doc(rows[::-1]),'epe_all'); np.testing.assert_array_equal(a,b)
        with self.assertRaises(ValueError): paired_arrays(doc(rows),doc(rows[:-1]),'epe_all')
        changed=[dict(r) for r in rows]; changed[0]['support_sha256']='wrong'
        with self.assertRaises(ValueError): paired_arrays(doc(rows),doc(changed),'epe_all')
    def test_unavailable_metric(self):
        self.assertIsNone(bootstrap(np.full(3,np.nan),np.full(3,np.nan),np.arange(3),1000,1))
    def test_single_cluster_fails(self):
        with self.assertRaises(ValueError): bootstrap(np.arange(3.),np.arange(3.),np.zeros(3),1000,1)
    def test_flow_roundtrips_and_zip_layout(self):
        import cv2,h5py
        f=np.zeros((3,4,2),np.float32); f[:,:,0]=1.125; f[:,:,1]=-3.25
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for ext in ('.flo','.flo5','.png'):
                p=root/('test'+ext); write_flow(p,f); verify_flow(p,(3,4))
                if ext=='.png':
                    a=cv2.imread(str(p),cv2.IMREAD_UNCHANGED)[:,:,::-1]
                    np.testing.assert_allclose((a[:,:,:2].astype(float)-32768)/64,f)
                elif ext=='.flo5':
                    with h5py.File(p) as h: np.testing.assert_equal(h['flow'][()],f)
                else:
                    np.testing.assert_equal(np.frombuffer(p.read_bytes()[12:],dtype='<f4').reshape(3,4,2),f)
            with self.assertRaises(ValueError): write_flow(root/'overflow.png',f+1000)
            for bench,rel in [('kitti2012','000000_10.png'),('kitti2015','flow/000000_10.png')]:
                write_flow(root/'pred'/rel,f)
                dest=pack(bench,root/'pred',root,[{'output':rel}])[0]
                with zipfile.ZipFile(dest) as z: self.assertEqual(z.namelist(),[rel])
    def test_spring_both_sides_both_directions(self):
        with tempfile.TemporaryDirectory() as td:
            for side in ('left','right'):
                d=Path(td)/'test'/'0047'/f'frame_{side}'; d.mkdir(parents=True)
                for i in (1,2,3): (d/f'frame_{side}_{i:04d}.png').touch()
            r=records('spring',td); self.assertEqual(len(r),8)
            back=[x for x in r if 'flow_BW_left_0002' in x['output']][0]
            self.assertTrue(back['image1'].endswith('0002.png')); self.assertTrue(back['image2'].endswith('0001.png'))

if __name__=='__main__': unittest.main()
