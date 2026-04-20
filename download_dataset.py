"""
download_dataset.py  —  NIFTY 500 Historical Data Downloader
Downloads 5 years of daily OHLCV data from Yahoo Finance and saves to dataset/
Run once before evaluate.py to build the local dataset.
"""

import os
import time
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import yfinance as yf

# ─── NIFTY 500 Tickers (NSE symbols with .NS suffix) ─────────────────────────
NIFTY500 = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","BAJFINANCE.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS",
    "ITC.NS","ASIANPAINT.NS","AXISBANK.NS","LT.NS","MARUTI.NS",
    "SUNPHARMA.NS","TITAN.NS","ULTRACEMCO.NS","NESTLEIND.NS","WIPRO.NS",
    "TECHM.NS","HCLTECH.NS","POWERGRID.NS","NTPC.NS","ONGC.NS",
    "TATAMOTORS.NS","TATASTEEL.NS","JSWSTEEL.NS","ADANIENT.NS","ADANIPORTS.NS",
    "BAJAJFINSV.NS","BPCL.NS","BRITANNIA.NS","CIPLA.NS","COALINDIA.NS",
    "DIVISLAB.NS","DRREDDY.NS","EICHERMOT.NS","GRASIM.NS","HEROMOTOCO.NS",
    "HINDALCO.NS","INDUSINDBK.NS","IOC.NS","M&M.NS","SBILIFE.NS",
    "APOLLOHOSP.NS","BAJAJ-AUTO.NS","HDFCLIFE.NS","PIDILITIND.NS","TATACONSUM.NS",
    "AMBUJACEM.NS","BANKBARODA.NS","BOSCHLTD.NS","CANBK.NS","CHOLAFIN.NS",
    "COLPAL.NS","DABUR.NS","DLF.NS","GAIL.NS","GODREJCP.NS",
    "HAVELLS.NS","ICICIPRULI.NS","ICICIGI.NS","IDFCFIRSTB.NS","INDHOTEL.NS",
    "INDUSTOWER.NS","IRCTC.NS","JINDALSTEL.NS","JUBLFOOD.NS","L&TFH.NS",
    "LICHSGFIN.NS","LUPIN.NS","MCDOWELL-N.NS","MARICO.NS","MOTHERSON.NS",
    "MPHASIS.NS","MRF.NS","NMDC.NS","OBEROIRLTY.NS","OFSS.NS",
    "PEL.NS","PERSISTENT.NS","PETRONET.NS","PFC.NS","PIIND.NS",
    "POLYCAB.NS","RECLTD.NS","SAIL.NS","SHREECEM.NS","SIEMENS.NS",
    "SRF.NS","TORNTPHARM.NS","UBL.NS","UNITDSPR.NS","UPL.NS",
    "VEDL.NS","VOLTAS.NS","WHIRLPOOL.NS","ZOMATO.NS","NYKAA.NS",
    "PAYTM.NS","DELHIVERY.NS","TATAPOWER.NS","ADANIGREEN.NS","ADANITRANS.NS",
    "APLAPOLLO.NS","AARTIIND.NS","AAVAS.NS","ABCAPITAL.NS","ABFRL.NS",
    "ACE.NS","AFFLE.NS","AJANTPHARM.NS","ALKEM.NS","ALKYLAMINE.NS",
    "AMARAJABAT.NS","AMBER.NS","ANALOGICX.NS","ANGELONE.NS","APLAPOLLO.NS",
    "APTUS.NS","ASTRAL.NS","ATUL.NS","AUROPHARMA.NS","AVANTIFEED.NS",
    "BAJAJHLDNG.NS","BALRAMCHIN.NS","BATAINDIA.NS","BAYERCROP.NS","BEML.NS",
    "BEL.NS","BERGEPAINT.NS","BHEL.NS","BLUESTARCO.NS","BRIGADE.NS",
    "CAMS.NS","CANFINHOME.NS","CARBORUNIV.NS","CASTROLIND.NS","CEATLTD.NS",
    "CENTURYTEX.NS","CERA.NS","CGPOWER.NS","CHAMBLFERT.NS","CLEAN.NS",
    "COFORGE.NS","CONCOR.NS","COROMANDEL.NS","CREDITACC.NS","CRISIL.NS",
    "CROMPTON.NS","CUMMINSIND.NS","CYIENT.NS","DATAPATTNS.NS","DCMSHRIRAM.NS",
    "DEEPAKFERT.NS","DEEPAKNTR.NS","DELTACORP.NS","DEVYANI.NS","DIXON.NS",
    "DMART.NS","EDELWEISS.NS","EMAMILTD.NS","ENDURANCE.NS","ENGINERSIN.NS",
    "EPIGRAL.NS","EQUITASBNK.NS","ESCORTS.NS","EXIDEIND.NS","FINEORG.NS",
    "FINPIPE.NS","FLUOROCHEM.NS","FORCEMOT.NS","FORTIS.NS","FRETAIL.NS",
    "GLAND.NS","GLAXO.NS","GMRINFRA.NS","GODFRYPHLP.NS","GODREJAGRO.NS",
    "GODREJIND.NS","GODREJPROP.NS","GRANULES.NS","GSPL.NS","HAPPSTMNDS.NS",
    "HATSUN.NS","HCG.NS","HDFCAMC.NS","HFCL.NS","HIKAL.NS",
    "HINDCOPPER.NS","HINDPETRO.NS","IDEAFORGE.NS","IEX.NS","IGPL.NS",
    "IIFL.NS","IILFL.NS","INDIAMART.NS","INDIANB.NS","INDIGO.NS",
    "INOXWIND.NS","INTELLECT.NS","IPL.NS","ISEC.NS","JAYNECOIND.NS",
    "JBCHEPHARM.NS","JKCEMENT.NS","JKLAKSHMI.NS","JKPAPER.NS","JMFINANCIL.NS",
    "JSWENERGY.NS","JTEKTINDIA.NS","KAJARIACER.NS","KALYANKJIL.NS","KANSAINER.NS",
    "KEC.NS","KFINTECH.NS","KINETIC.NS","KPITTECH.NS","KRBL.NS",
    "KSCL.NS","LALPATHLAB.NS","LAURUSLABS.NS","LAXMIMACH.NS","LINDEINDIA.NS",
    "LXCHEM.NS","MAHINDCIE.NS","MAHLIFE.NS","MANAPPURAM.NS","MASFIN.NS",
    "MAXHEALTH.NS","MCX.NS","MEDPLUS.NS","METROBRAND.NS","MFSL.NS",
    "MINDTREE.NS","MIRZAINT.NS","MODI.NS","MUTHOOTFIN.NS","NAM-INDIA.NS",
    "NATIONALUM.NS","NAUKRI.NS","NAVINFLUOR.NS","NESCO.NS","NETWORK18.NS",
    "NLCINDIA.NS","NOCIL.NS","NUVOCO.NS","NYKAA.NS","OLECTRA.NS",
    "ORIENTCEM.NS","PAGEIND.NS","PARAS.NS","PCBL.NS","PGHH.NS",
    "PHOENIXLTD.NS","PNBHOUSING.NS","PRABHAT.NS","PRASOL.NS","PRSMJOHNSN.NS",
    "PSPPROJECT.NS","PVR.NS","RADICO.NS","RAIN.NS","RAJESHEXPO.NS",
    "RALLIS.NS","RAMCOCEM.NS","RBLBANK.NS","REDINGTON.NS","RELAXO.NS",
    "RITES.NS","RKFORGE.NS","ROSSARI.NS","ROUTE.NS","SAFARI.NS",
    "SANDUMA.NS","SANOFI.NS","SAPPHIRE.NS","SCHAEFFLER.NS","SEQUENT.NS",
    "SHYAMMETL.NS","SJVN.NS","SOBHA.NS","SOLARA.NS","SOLARINDS.NS",
    "SONACOMS.NS","SONA.NS","SPANDANA.NS","SPARC.NS","STCINDIA.NS",
    "SUNTV.NS","SUPRAJIT.NS","SURYAROSE.NS","SUVEN.NS","SUVENPHAR.NS",
    "SYMPHONY.NS","TANLA.NS","TATACHEM.NS","TATAINVEST.NS","TATACOMM.NS",
    "TATAELXSI.NS","TATAMETALI.NS","TCNS.NS","TEAMLEASE.NS","TIINDIA.NS",
    "TIMKEN.NS","TRENT.NS","TRIDENT.NS","TRIVENI.NS","UCOBANK.NS",
    "UJJIVANSFB.NS","UNIINFO.NS","UNIONBANK.NS","V2RETAIL.NS","VAIBHAVGBL.NS",
    "VARROC.NS","VGUARD.NS","VIP.NS","VMART.NS","VSTIND.NS",
    "WABAG.NS","WELCORP.NS","WESTLIFE.NS","WHIRLPOOL.NS","WIPRO.NS",
    "XCHANGING.NS","YESBANK.NS","ZEEL.NS","ZENTEC.NS","ZYDUSLIFE.NS",
]

# Deduplicate while preserving order
seen = set()
NIFTY500 = [t for t in NIFTY500 if not (t in seen or seen.add(t))]

OUT_DIR  = "dataset"
OUT_CSV  = os.path.join(OUT_DIR, "nse_500_5years.csv")
TICK_CSV = os.path.join(OUT_DIR, "nifty500_tickers.csv")
START    = "2020-01-01"
END      = "2025-01-01"
BATCH    = 10          # download in batches to be polite to the API

os.makedirs(OUT_DIR, exist_ok=True)

# ─── Save ticker list ─────────────────────────────────────────────────────────
pd.DataFrame({"Ticker": NIFTY500}).to_csv(TICK_CSV, index=False)

print()
print("=" * 68)
print("   NIFTY 500  —  Historical Dataset Downloader")
print("=" * 68)
print(f"   Tickers  : {len(NIFTY500)}")
print(f"   Period   : {START}  to  {END}  (5 years daily)")
print(f"   Output   : {OUT_CSV}")
print("=" * 68)
print()

all_frames = []
failed     = []

for batch_start in range(0, len(NIFTY500), BATCH):
    batch = NIFTY500[batch_start : batch_start + BATCH]
    for ticker in batch:
        try:
            df = yf.download(ticker, start=START, end=END,
                             interval="1d", progress=False, auto_adjust=True)
            if df.empty:
                failed.append(ticker)
                continue
            # Flatten MultiIndex columns if yfinance returned them
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            df["Ticker"] = ticker
            df.reset_index(inplace=True)
            df.rename(columns={"index": "Date", "Price": "Date"}, errors="ignore")
            all_frames.append(df)
            done = batch_start + batch.index(ticker) + 1
            pct  = done / len(NIFTY500) * 100
            print(f"  [{done:>3}/{len(NIFTY500)}]  {pct:>5.1f}%   {ticker:<22s}  rows={len(df)}")
        except Exception as e:
            failed.append(ticker)
            print(f"  [SKIP]  {ticker:<22s}  error: {e}")
    time.sleep(0.5)   # gentle throttle between batches

print()
if not all_frames:
    print("  [ERROR] No data downloaded. Check your internet connection.")
    raise SystemExit(1)

final_df = pd.concat(all_frames, ignore_index=True)
final_df.to_csv(OUT_CSV, index=False)

print("=" * 68)
print(f"  [DONE]  Saved {len(final_df):,} rows  x  {final_df.shape[1]} columns")
print(f"          Stocks downloaded : {len(all_frames)}")
print(f"          Stocks failed     : {len(failed)}  {failed[:5] if failed else ''}")
print(f"          File size         : {os.path.getsize(OUT_CSV)/1e6:.1f} MB")
print(f"          Saved to          : {OUT_CSV}")
print("=" * 68)
print()
