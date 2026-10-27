import sys, unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from router import build_model,route,train

class RouterTests(unittest.TestCase):
    def test_abstains_on_ambiguous_text(self):
        df=pd.DataFrame({"text":["card payment failed","card payment issue","change my pin","reset the pin","transfer money","send a transfer"],"label":["card","card","pin","pin","transfer","transfer"]})
        model=build_model().fit(df.text,df.label)
        result=route(model,"hello",threshold=1.0)
        self.assertTrue(result["needs_human_review"]); self.assertIsNone(result["category"])
    def test_confident_route_has_contract(self):
        df=pd.DataFrame({"text":["card payment failed","card payment issue","change my pin","reset the pin","transfer money","send a transfer"],"label":["card","card","pin","pin","transfer","transfer"]})
        model=build_model().fit(df.text,df.label)
        result=route(model,"card payment failed",threshold=0.0)
        self.assertIn("confidence",result); self.assertFalse(result["needs_human_review"])
    def test_empty_text_rejected(self):
        with self.assertRaises(ValueError): route(None," ")

if __name__=="__main__": unittest.main()
