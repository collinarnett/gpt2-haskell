{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module GPT2.Loader where

import Control.Monad (zipWithM)
import Control.Monad.IO.Class (liftIO)
import Control.Monad.Trans.Maybe
import GHC.Exts (IsList (fromList))
import GHC.Utils.Monad (zipWith3M)
import GPT2.Model
import SafeTensors
import qualified Torch as UT
import qualified Torch.DType as D
import Torch.HList
import Torch.Typed hiding
  ( MultiheadAttention,
    TransformerLayer,
    TransformerMLP,
    keys,
    transformerLM,
  )
import Prelude hiding (lookup)

type ModelDevice = '(CPU, 0)

type MaxSeqLen = 1024

type NumAttnLayers = 12

type NumHeads = 12

type FFNDim = 3072

type PaddingIdx = 0

type NumEmbeds = 768

type VocabSize = 50257

type Model =
  GPT2
    NumAttnLayers
    NumHeads
    FFNDim
    PaddingIdx
    MaxSeqLen
    VocabSize
    NumEmbeds
    D.Float
    ModelDevice

type ModelSpec =
  GPT2Spec
    NumAttnLayers
    NumHeads
    FFNDim
    PaddingIdx
    MaxSeqLen
    VocabSize
    NumEmbeds
    D.Float
    ModelDevice

mkLN ::
  UT.Tensor ->
  UT.Tensor ->
  Double ->
  IO (LayerNorm '[NumEmbeds] D.Float ModelDevice)
mkLN lnw lnb e = do
  weights' <- makeIndependent $ UnsafeMkTensor @ModelDevice @D.Float @'[NumEmbeds] lnw
  bias' <- makeIndependent $ UnsafeMkTensor @ModelDevice @D.Float @'[NumEmbeds] lnb
  return $ LayerNorm weights' bias' e

mkQKVLinear ::
  UT.Tensor ->
  UT.Tensor ->
  IO (Linear NumEmbeds NumEmbeds D.Float ModelDevice)
mkQKVLinear w b = do
  --  _ <- print $ "QKV Linear untyped weight shape is " ++ show (UT.shape w)
  --  _ <- print $ "QKV Linear untyped bias shape is " ++ show (UT.shape b)
  weights' <- makeIndependent $ UnsafeMkTensor @ModelDevice @D.Float @'[NumEmbeds, NumEmbeds] w
  bias' <- makeIndependent $ UnsafeMkTensor @ModelDevice @D.Float @'[NumEmbeds] b
  return $ Linear weights' bias'

mkOutputLinear ::
  UT.Tensor ->
  UT.Tensor ->
  IO (Linear NumEmbeds VocabSize D.Float ModelDevice)
mkOutputLinear w b = do
  -- _ <- print $ "Output Linear untyped weight shape is " ++ show (UT.shape w)
  -- _ <- print $ "Output Linear untyped bias shape is " ++ show (UT.shape b)
  weights' <- makeIndependent $ UnsafeMkTensor @ModelDevice @D.Float @'[VocabSize, NumEmbeds] w
  bias' <- makeIndependent $ UnsafeMkTensor @ModelDevice @D.Float @'[VocabSize] b
  return $ Linear weights' bias'

mkLinear ::
  UT.Tensor ->
  UT.Tensor ->
  IO (Linear FFNDim NumEmbeds D.Float ModelDevice)
mkLinear w b = do
  let transposedW = UT.transpose (UT.Dim 1) (UT.Dim 0) w
  -- _ <- print $ "Linear untyped weight shape is " ++ show (UT.shape transposedW)
  -- _ <- print $ "Linear untyped bias shape is " ++ show (UT.shape b)
  weights' <- makeIndependent $ UnsafeMkTensor @ModelDevice @D.Float @'[NumEmbeds, FFNDim] $ transposedW
  bias' <- makeIndependent $ UnsafeMkTensor @ModelDevice @D.Float @'[NumEmbeds] b
  return $ Linear weights' bias'

mkLinearProj ::
  UT.Tensor ->
  UT.Tensor ->
  IO (Linear NumEmbeds FFNDim D.Float ModelDevice)
mkLinearProj w b = do
  let transposedW = UT.transpose (UT.Dim 1) (UT.Dim 0) w
  -- _ <- print $ "LinearProj untyped weight shape is " ++ show (UT.shape transposedW)
  -- _ <- print $ "LinearProj untyped bias shape is " ++ show (UT.shape b)
  weights' <- makeIndependent $ UnsafeMkTensor @ModelDevice @D.Float @'[FFNDim, NumEmbeds] transposedW
  bias' <- makeIndependent $ UnsafeMkTensor @ModelDevice @D.Float @'[FFNDim] b
  return $ Linear weights' bias'

mkAttn ::
  UT.Tensor ->
  UT.Tensor ->
  UT.Tensor ->
  UT.Tensor ->
  IO (MultiheadAttention NumEmbeds NumHeads D.Float ModelDevice)
mkAttn w1 b1 w2 b2 = do
  -- _ <- print $ "Attention weight1 untyped weight shape is " ++ show (UT.shape w1)
  -- _ <- print $ "Attention bias1 untyped bias shape is " ++ show (UT.shape b1)
  let weightSplits = UT.split 768 (UT.Dim 0) (UT.transpose (UT.Dim 0) (UT.Dim 1) w1)
      biasSplits = UT.split 768 (UT.Dim 0) b1
  -- _ <- print $ "weightSplits untyped shape is " ++ show (UT.shape $ head weightSplits)
  -- _ <- print $ "biasSplits untyped shape is " ++ show (UT.shape $ head biasSplits)
  -- _ <- print $ "weightSplits length is " ++ show (Prelude.length weightSplits)
  inProj <- zipWithM mkQKVLinear weightSplits biasSplits
  outProj <- mkQKVLinear (UT.transpose (UT.Dim 0) (UT.Dim 1) w2) b2
  return $ MultiheadAttention (head inProj) (inProj !! 1) (inProj !! 2) outProj

mkMLP ::
  UT.Tensor ->
  UT.Tensor ->
  UT.Tensor ->
  UT.Tensor ->
  UT.Tensor ->
  UT.Tensor ->
  IO (TransformerMLP NumEmbeds FFNDim D.Float ModelDevice)
mkMLP w1 b1 w2 b2 lnw lnb =
  TransformerMLP
    <$> mkLinearProj w1 b1
    <*> mkLinear w2 b2
    <*> mkLN lnw lnb 0.00001

mkPosEmbeddings ::
  UT.Tensor ->
  IO (Embedding Nothing MaxSeqLen NumEmbeds 'Constant D.Float ModelDevice)
mkPosEmbeddings = return . ConstEmbedding . UnsafeMkTensor @ModelDevice @D.Float @'[MaxSeqLen, NumEmbeds]

mkTokEmbeddings ::
  UT.Tensor ->
  IO (Embedding (Just PaddingIdx) VocabSize NumEmbeds Learned D.Float ModelDevice)
mkTokEmbeddings tensor = do
  param <- makeIndependent $ UnsafeMkTensor @ModelDevice @D.Float @'[VocabSize, NumEmbeds] tensor
  return $ LearnedEmbedding param

mkTransformerLayer ::
  MultiheadAttention NumEmbeds NumHeads D.Float ModelDevice ->
  LayerNorm '[NumEmbeds] D.Float ModelDevice ->
  TransformerMLP NumEmbeds FFNDim D.Float ModelDevice ->
  IO (TransformerLayer NumEmbeds NumHeads FFNDim D.Float ModelDevice)
mkTransformerLayer attn ln' mlp = return $ TransformerLayer attn ln' mlp

getTransformerLayerNorms ::
  SafeTensors ->
  Int ->
  MaybeT IO (LayerNorm '[NumEmbeds] D.Float ModelDevice)
getTransformerLayerNorms weights' i =
  liftIO
    =<< mkLN
      <$> hoistMaybe (lookupWithSafeTensorKey ("h." ++ show i ++ ".ln_1.weight") weights')
      <*> hoistMaybe (lookupWithSafeTensorKey ("h." ++ show i ++ ".ln_1.bias") weights')
      <*> pure 0.00001

getAttnParameters ::
  SafeTensors ->
  Int ->
  MaybeT IO (MultiheadAttention NumEmbeds NumHeads D.Float ModelDevice)
getAttnParameters weights' i =
  liftIO
    =<< mkAttn
      <$> hoistMaybe (lookupWithSafeTensorKey ("h." ++ show i ++ ".attn.c_attn.weight") weights')
      <*> hoistMaybe (lookupWithSafeTensorKey ("h." ++ show i ++ ".attn.c_attn.bias") weights')
      <*> hoistMaybe (lookupWithSafeTensorKey ("h." ++ show i ++ ".attn.c_proj.weight") weights')
      <*> hoistMaybe (lookupWithSafeTensorKey ("h." ++ show i ++ ".attn.c_proj.bias") weights')

getMLPParameters ::
  SafeTensors ->
  Int ->
  MaybeT IO (TransformerMLP NumEmbeds FFNDim D.Float ModelDevice)
getMLPParameters weights' i = do
  liftIO
    =<< mkMLP
      <$> hoistMaybe (lookupWithSafeTensorKey ("h." ++ show i ++ ".mlp.c_fc.weight") weights')
      <*> hoistMaybe (lookupWithSafeTensorKey ("h." ++ show i ++ ".mlp.c_fc.bias") weights')
      <*> hoistMaybe (lookupWithSafeTensorKey ("h." ++ show i ++ ".mlp.c_proj.weight") weights')
      <*> hoistMaybe (lookupWithSafeTensorKey ("h." ++ show i ++ ".mlp.c_proj.bias") weights')
      <*> hoistMaybe (lookupWithSafeTensorKey ("h." ++ show i ++ ".ln_2.weight") weights')
      <*> hoistMaybe (lookupWithSafeTensorKey ("h." ++ show i ++ ".ln_2.bias") weights')

loadGPT2FromSafeTensors ::
  SafeTensors ->
  MaybeT
    IO
    ( GPT2
        NumAttnLayers
        NumHeads
        FFNDim
        PaddingIdx
        MaxSeqLen
        VocabSize
        NumEmbeds
        D.Float
        ModelDevice
    )
loadGPT2FromSafeTensors st = do
  tokenEmbeddings <- hoistMaybe $ lookupWithSafeTensorKey "wte.weight" st
  posEmbeddings <- hoistMaybe $ lookupWithSafeTensorKey "wpe.weight" st
  mlpLayers <- mapM (getMLPParameters st) [11, 10 .. 0]
  attnLayers <- mapM (getAttnParameters st) [11, 10 .. 0]
  transformerLayerNorms <- mapM (getTransformerLayerNorms st) [11, 10 .. 0]
  finalLayerNormWeight <- hoistMaybe $ lookupWithSafeTensorKey "ln_f.weight" st
  finalLayerNormBias <- hoistMaybe $ lookupWithSafeTensorKey "ln_f.bias" st
  transformerLayers <- liftIO $ zipWith3M mkTransformerLayer attnLayers transformerLayerNorms mlpLayers
  let layers = (fromList transformerLayers :: Maybe (HList (HReplicateR NumHeads (TransformerLayer NumEmbeds NumHeads FFNDim D.Float ModelDevice))))
  sharedParams <- liftIO $ mkTokEmbeddings tokenEmbeddings
  zeroBias <- liftIO $ makeIndependent $ zeros @'[VocabSize] @'D.Float @ModelDevice
  GPT2 sharedParams
    <$> liftIO (mkPosEmbeddings posEmbeddings)
    <*> hoistMaybe layers
    <*> liftIO (mkLN finalLayerNormWeight finalLayerNormBias 0.00001)
    <*> return (Linear (learnedEmbedWeights sharedParams) zeroBias)
