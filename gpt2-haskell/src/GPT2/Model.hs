{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE Strict #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=0 #-}

module GPT2.Model where

import Control.Arrow (arr, (>>>))
import Control.Category (id)
import Data.Proxy
import Data.Vector.Sized (Vector)
import GHC.Generics
import GHC.TypeLits
import GPT2.Torch.Typed.Functional (geluApproximate)
import System.IO.Unsafe (unsafePerformIO)
import qualified Torch.DType as D
import qualified Torch.Device as D
import Torch.HList
import Torch.NN (HasForward (..))
import qualified Torch.NN as A
import qualified Torch.Tensor as UT
import Torch.Typed.Auxiliary
import Torch.Typed.Factories
import Torch.Typed.Functional hiding (linear, log, trace)
import Torch.Typed.NN.Arrow (Net (..), layer, residual)
import Torch.Typed.NN.Linear
import Torch.Typed.NN.Normalization
import Torch.Typed.NN.Sparse
import Torch.Typed.Parameter
import Torch.Typed.Representable (tabulateList)
import Torch.Typed.Tensor
import Prelude hiding (cos, exp, id, sin)

--------------------------------------------------------------------------------
-- Relation-Aware Multi-Headed Attention Layer
--------------------------------------------------------------------------------

data
  MultiheadAttentionSpec
    (numEmbeds :: Nat)
    (numHeads :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  MultiheadAttentionSpec ::
    -- | spec for dropout
    MultiheadAttentionSpec numEmbeds numHeads dtype device
  deriving (Show, Eq)

data
  MultiheadAttention
    (numEmbeds :: Nat)
    (numHeads :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  MultiheadAttention ::
    { -- | packed in-projection for q, k, v
      mhaQInProj :: Linear numEmbeds numEmbeds dtype device,
      -- | in-projection for key
      mhaKInProj :: Linear numEmbeds numEmbeds dtype device,
      -- | in-projection for value
      mhaVInProj :: Linear numEmbeds numEmbeds dtype device,
      -- | out-projection
      mhaOutProj :: Linear numEmbeds numEmbeds dtype device
    } ->
    MultiheadAttention numEmbeds numHeads dtype device
  deriving (Show, Generic, Parameterized)

multiheadAttention ::
  forall numEmbeds numHeads inputSeqLen batchSize headDim dtype device.
  ( 1 <= numHeads,
    numEmbeds ~ (headDim * numHeads),
    All KnownNat '[numEmbeds, numEmbeds, numEmbeds, numHeads, inputSeqLen, batchSize, headDim],
    KnownDType dtype,
    StandardFloatingPointDTypeValidation device dtype,
    MatMulDTypeIsValid device dtype,
    BasicArithmeticDTypeIsValid device dtype,
    dtype ~ SumDType dtype,
    SumDTypeIsValid device dtype,
    KnownDevice device
  ) =>
  -- | multi-head attention model ADT
  MultiheadAttention numEmbeds numHeads dtype device ->
  -- | optional attention mask, broadcast over batch and head
  Maybe (Tensor device dtype '[inputSeqLen, inputSeqLen]) ->
  -- | query representation
  Tensor device dtype '[batchSize, inputSeqLen, numEmbeds] ->
  -- | key representation
  Tensor device dtype '[batchSize, inputSeqLen, numEmbeds] ->
  -- | value representation
  Tensor device dtype '[batchSize, inputSeqLen, numEmbeds] ->
  -- | attention and attention averaged over heads
  IO (Tensor device dtype '[batchSize, inputSeqLen, numEmbeds])
multiheadAttention MultiheadAttention {..} attentionMask query key value = do
  let weights =
        softmax @3
          . _maskAttention
          $ _attentionWeights
  return $ _attention weights
  where
    _attentionWeights =
      let scaling = Prelude.sqrt . fromIntegral $ natValI @headDim :: Double
          q = reshape' . forward mhaQInProj $ query
          k = reshape' . forward mhaKInProj $ key
          weights = divScalar scaling $ matmul q (transpose @2 @3 k)
       in weights
    _maskAttention attentionWeights =
      case attentionMask of
        Nothing -> attentionWeights
        Just am -> attentionWeights `add` am
    _attention attentionWeights =
      let v = reshape' . forward mhaVInProj $ value
          attention = transpose @1 @2 $ matmul attentionWeights v
       in forward mhaOutProj . reshape @'[batchSize, inputSeqLen, numEmbeds] $ attention
    reshape' ::
      forall inputSeqLen'.
      (KnownNat inputSeqLen') =>
      Tensor device dtype '[batchSize, inputSeqLen', numEmbeds] ->
      Tensor device dtype '[batchSize, numHeads, inputSeqLen', headDim]
    reshape' t' = transpose @1 @2 $ reshape @'[batchSize, inputSeqLen', numHeads, headDim] t'

-- | Self-attention as an arrow: the one input is fanned out to query, key and
-- value.
selfAttention ::
  forall numEmbeds numHeads inputSeqLen batchSize headDim dtype device.
  ( 1 <= numHeads,
    numEmbeds ~ (headDim * numHeads),
    All KnownNat '[numEmbeds, numHeads, inputSeqLen, batchSize, headDim],
    KnownDType dtype,
    StandardFloatingPointDTypeValidation device dtype,
    MatMulDTypeIsValid device dtype,
    BasicArithmeticDTypeIsValid device dtype,
    dtype ~ SumDType dtype,
    SumDTypeIsValid device dtype,
    KnownDevice device
  ) =>
  MultiheadAttention numEmbeds numHeads dtype device ->
  Maybe (Tensor device dtype '[inputSeqLen, inputSeqLen]) ->
  Net
    (Tensor device dtype '[batchSize, inputSeqLen, numEmbeds])
    (Tensor device dtype '[batchSize, inputSeqLen, numEmbeds])
selfAttention mha attentionMask = Net $ \x -> multiheadAttention mha attentionMask x x x

instance
  ( All KnownNat '[numEmbeds, numEmbeds, numEmbeds, numHeads],
    KnownDType dtype,
    KnownDevice device,
    RandDTypeIsValid device dtype
  ) =>
  A.Randomizable
    (MultiheadAttentionSpec numEmbeds numHeads dtype device)
    (MultiheadAttention numEmbeds numHeads dtype device)
  where
  sample MultiheadAttentionSpec =
    MultiheadAttention
      <$> A.sample LinearSpec
      <*> A.sample LinearSpec
      <*> A.sample LinearSpec
      <*> A.sample LinearSpec

--------------------------------------------------------------------------------
-- Transformer MLP Layer
--------------------------------------------------------------------------------

data
  TransformerMLPSpec
    (numEmbeds :: Nat)
    (ffnDim :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  TransformerMLPSpec ::
    forall numEmbeds ffnDim dtype device.
    { -- | epsilon for layer norm
      epsSpec :: Double
    } ->
    TransformerMLPSpec numEmbeds ffnDim dtype device
  deriving (Show, Eq)

data
  TransformerMLP
    (numEmbeds :: Nat)
    (ffnDim :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  TransformerMLP ::
    forall numEmbeds ffnDim dtype device.
    { -- | first fully connected layer
      linear0 :: Linear numEmbeds ffnDim dtype device,
      -- | second fully connected layer
      linear1 :: Linear ffnDim numEmbeds dtype device,
      -- | layer norm
      ln :: LayerNorm '[numEmbeds] dtype device
    } ->
    TransformerMLP numEmbeds ffnDim dtype device
  deriving (Show, Generic, Parameterized)

-- | @x + FF(LN x)@, where @FF@ is the two-layer feed-forward network.
transformerMLP ::
  forall numEmbeds ffnDim maxSeqLen batchSize dtype device.
  ( BasicArithmeticDTypeIsValid device dtype,
    StandardFloatingPointDTypeValidation device dtype,
    KnownNat numEmbeds,
    KnownDevice device,
    GeluDTypeIsValid device dtype,
    IsSuffixOf '[numEmbeds] '[maxSeqLen, batchSize, numEmbeds]
  ) =>
  -- | MLP model ADT for transformer
  TransformerMLP numEmbeds ffnDim dtype device ->
  Net
    (Tensor device dtype '[maxSeqLen, batchSize, numEmbeds])
    (Tensor device dtype '[maxSeqLen, batchSize, numEmbeds])
transformerMLP TransformerMLP {..} =
  residual $
    layer ln
      >>> layer linear0
      >>> arr (`geluApproximate` "tanh")
      >>> layer linear1

instance
  ( All KnownNat '[numEmbeds, ffnDim],
    KnownDType dtype,
    KnownDevice device,
    RandDTypeIsValid device dtype
  ) =>
  A.Randomizable
    (TransformerMLPSpec numEmbeds ffnDim dtype device)
    (TransformerMLP numEmbeds ffnDim dtype device)
  where
  sample TransformerMLPSpec {..} =
    TransformerMLP
      <$> A.sample LinearSpec
      <*> A.sample LinearSpec
      <*> A.sample (LayerNormSpec epsSpec)

--------------------------------------------------------------------------------
-- Relation-Aware Transformer Layer
--------------------------------------------------------------------------------

data
  TransformerLayerSpec
    (numEmbeds :: Nat)
    (numHeads :: Nat)
    (ffnDim :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  TransformerLayerSpec ::
    forall numEmbeds numHeads ffnDim dtype device.
    { mhaSpec :: MultiheadAttentionSpec numEmbeds numHeads dtype device,
      epsSpec' :: Double,
      mlpSpec :: TransformerMLPSpec numEmbeds ffnDim dtype device
    } ->
    TransformerLayerSpec numEmbeds numHeads ffnDim dtype device
  deriving (Show, Eq)

data
  TransformerLayer
    (numEmbeds :: Nat)
    (numHeads :: Nat)
    (ffnDim :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  TransformerLayer ::
    forall numEmbeds numHeads ffnDim dtype device.
    { -- | multi-head attention
      transformerLayer_mha :: MultiheadAttention numEmbeds numHeads dtype device,
      -- | layer norm
      transformerLayer_ln :: LayerNorm '[numEmbeds] dtype device,
      -- | MLP
      transformerLayer_mlp :: TransformerMLP numEmbeds ffnDim dtype device
    } ->
    TransformerLayer numEmbeds numHeads ffnDim dtype device
  deriving (Show, Generic, Parameterized)

-- | A pre-norm decoder block: attention and feed-forward, each wrapped in a
-- skip connection.
--
-- > x'  = x  + Attend(LN₁ x)
-- > x'' = x' + FF(LN₂ x')
transformerLayer ::
  forall (numHeads :: Nat) (ffnDim :: Nat) (numEmbeds :: Nat) (headDim :: Nat) (inputSeqLen :: Nat) (batchSize :: Nat) dtype device.
  ( 1 <= numHeads,
    numEmbeds ~ (headDim * numHeads),
    All KnownNat '[numEmbeds, numHeads, inputSeqLen, batchSize, headDim],
    IsSuffixOf '[numEmbeds] '[batchSize, inputSeqLen, numEmbeds],
    KnownDType dtype,
    dtype ~ SumDType dtype,
    StandardFloatingPointDTypeValidation device dtype,
    GeluDTypeIsValid device dtype,
    MatMulDTypeIsValid device dtype,
    BasicArithmeticDTypeIsValid device dtype,
    SumDTypeIsValid device dtype,
    KnownDevice device
  ) =>
  -- | transformer layer model ADT
  TransformerLayer numEmbeds numHeads ffnDim dtype device ->
  -- | optional attention mask
  Maybe (Tensor device dtype '[inputSeqLen, inputSeqLen]) ->
  -- | transformer layer as a network
  Net
    (Tensor device dtype '[batchSize, inputSeqLen, numEmbeds])
    (Tensor device dtype '[batchSize, inputSeqLen, numEmbeds])
transformerLayer TransformerLayer {..} attentionMask =
  residual (layer transformerLayer_ln >>> selfAttention transformerLayer_mha attentionMask)
    >>> transformerMLP transformerLayer_mlp

instance
  ( All KnownNat '[numEmbeds, numEmbeds, numEmbeds, numHeads, ffnDim],
    KnownDType dtype,
    KnownDevice device,
    RandDTypeIsValid device dtype
  ) =>
  A.Randomizable
    (TransformerLayerSpec numEmbeds numHeads ffnDim dtype device)
    (TransformerLayer numEmbeds numHeads ffnDim dtype device)
  where
  sample TransformerLayerSpec {..} =
    TransformerLayer
      <$> A.sample mhaSpec
      <*> A.sample (LayerNormSpec epsSpec')
      <*> A.sample mlpSpec

--------------------------------------------------------------------------------
-- Transformer Language Model (GPT-2)
--------------------------------------------------------------------------------

data
  GPT2Spec
    (numAttnLayers :: Nat)
    (numHeads :: Nat)
    (ffnDim :: Nat)
    (paddingIdx :: Nat)
    (maxSeqLen :: Nat)
    (vocabSize :: Nat)
    (numEmbeds :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  GPT2Spec ::
    forall numAttnLayers numHeads ffnDim paddingIdx maxSeqLen vocabSize numEmbeds dtype device.
    { -- | spec for each and every transformer layer
      lmLayerSpec :: TransformerLayerSpec numEmbeds numHeads ffnDim dtype device,
      epsSpec'' :: Double
    } ->
    GPT2Spec numAttnLayers numHeads ffnDim paddingIdx maxSeqLen vocabSize numEmbeds dtype device
  deriving (Show, Eq)

data
  GPT2
    (numAttnLayers :: Nat)
    (numHeads :: Nat)
    (ffnDim :: Nat)
    (paddingIdx :: Nat)
    (maxSeqLen :: Nat)
    (vocabSize :: Nat)
    (numEmbeds :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  where
  GPT2 ::
    forall numAttnLayers numHeads ffnDim paddingIdx maxSeqLen vocabSize numEmbeds dtype device.
    { -- | token embedding
      tEmbedding :: Embedding ('Just paddingIdx) vocabSize numEmbeds 'Learned dtype device,
      -- | positional embedding
      tPosEmbedding :: Embedding 'Nothing maxSeqLen numEmbeds 'Constant dtype device,
      -- | transformer layers, in the order they are applied
      tLayers :: HList (TransformerLayers numAttnLayers numEmbeds numHeads ffnDim dtype device),
      -- | final layer norm
      tFinalLN :: LayerNorm '[numEmbeds] dtype device,
      -- | final output projection
      tProj :: Linear numEmbeds vocabSize dtype device
    } ->
    GPT2 numAttnLayers numHeads ffnDim paddingIdx maxSeqLen vocabSize numEmbeds dtype device
  deriving (Generic)

-- | The stack of identically shaped decoder blocks.
type TransformerLayers numAttnLayers numEmbeds numHeads ffnDim dtype device =
  HReplicateR numAttnLayers (TransformerLayer numEmbeds numHeads ffnDim dtype device)

deriving instance
  ( Show
      ( HList
          ( TransformerLayers
              numAttnLayers
              numEmbeds
              numHeads
              ffnDim
              dtype
              device
          )
      )
  ) =>
  Show (GPT2 numAttnLayers numHeads ffnDim paddingIdx maxSeqLen vocabSize numEmbeds dtype device)

instance
  ( layers ~ TransformerLayers numAttnLayers numEmbeds numHeads ffnDim dtype device,
    Parameterized
      ( HList
          layers
      ),
    HAppendFD
      (Parameters (HList layers))
      '[ Parameter device dtype '[numEmbeds],
         Parameter device dtype '[numEmbeds],
         Parameter device dtype '[vocabSize, numEmbeds],
         Parameter device dtype '[vocabSize]
       ]
      ( Parameters (HList layers)
          ++ '[ Parameter device dtype '[numEmbeds],
                Parameter device dtype '[numEmbeds],
                Parameter device dtype '[vocabSize, numEmbeds],
                Parameter device dtype '[vocabSize]
              ]
      )
  ) =>
  Parameterized (GPT2 numAttnLayers numHeads ffnDim paddingIdx maxSeqLen vocabSize numEmbeds dtype device)

-- | @M[i, j] = 0@ where position @i@ may attend to position @j@ and @-inf@
-- where it may not: the causal mask as the formula that defines it.
causalMask ::
  forall inputSeqLen dtype device.
  ( KnownNat inputSeqLen,
    TensorOptions '[inputSeqLen, inputSeqLen] dtype device,
    UT.TensorLike (ComputeHaskellType dtype),
    Fractional (ComputeHaskellType dtype)
  ) =>
  Tensor device dtype '[inputSeqLen, inputSeqLen]
causalMask = toUnnamed (tabulateList @'[Vector inputSeqLen, Vector inputSeqLen] @dtype @device mask)
  where
    mask [i, j] = if j <= i then 0 else -1 / 0
    mask _ = 0

-- | Folds the layer stack into a single network by composing the blocks in
-- order.
data
  ComposeLayers
    (batchSize :: Nat)
    (inputSeqLen :: Nat)
    (dtype :: D.DType)
    (device :: (D.DeviceType, Nat))
  = ComposeLayers
  { -- | optional attention mask
    clAttentionMask :: Maybe (Tensor device dtype '[inputSeqLen, inputSeqLen])
  }

instance
  ( 1 <= numHeads,
    numEmbeds ~ (headDim * numHeads),
    All KnownNat '[numEmbeds, numHeads, inputSeqLen, batchSize, headDim],
    IsSuffixOf '[numEmbeds] '[batchSize, inputSeqLen, numEmbeds],
    KnownDType dtype,
    StandardFloatingPointDTypeValidation device dtype,
    MatMulDTypeIsValid device dtype,
    BasicArithmeticDTypeIsValid device dtype,
    GeluDTypeIsValid device dtype,
    dtype ~ SumDType dtype,
    SumDTypeIsValid device dtype,
    KnownDevice device
  ) =>
  Apply'
    (ComposeLayers batchSize inputSeqLen dtype device)
    ( TransformerLayer numEmbeds numHeads ffnDim dtype device,
      Net
        (Tensor device dtype '[batchSize, inputSeqLen, numEmbeds])
        (Tensor device dtype '[batchSize, inputSeqLen, numEmbeds])
    )
    ( Net
        (Tensor device dtype '[batchSize, inputSeqLen, numEmbeds])
        (Tensor device dtype '[batchSize, inputSeqLen, numEmbeds])
    )
  where
  apply' ComposeLayers {..} (layer', rest) = transformerLayer layer' clAttentionMask >>> rest

-- | The constraint that the layer stack can be folded into one network.
type ComposableLayers numAttnLayers numEmbeds numHeads ffnDim inputSeqLen batchSize dtype device =
  HFoldr
    (ComposeLayers batchSize inputSeqLen dtype device)
    ( Net
        (Tensor device dtype '[batchSize, inputSeqLen, numEmbeds])
        (Tensor device dtype '[batchSize, inputSeqLen, numEmbeds])
    )
    (TransformerLayers numAttnLayers numEmbeds numHeads ffnDim dtype device)
    ( Net
        (Tensor device dtype '[batchSize, inputSeqLen, numEmbeds])
        (Tensor device dtype '[batchSize, inputSeqLen, numEmbeds])
    )

transformerLM ::
  forall
    numAttnLayers
    numHeads
    ffnDim
    paddingIdx
    vocabSize
    numEmbeds
    inputSeqLen
    maxSeqLen
    batchSize
    dtype
    device.
  ( All KnownNat '[paddingIdx, numEmbeds, inputSeqLen, batchSize],
    IsSuffixOf '[numEmbeds] '[batchSize, inputSeqLen, numEmbeds],
    paddingIdx + 1 <= vocabSize,
    1 <= inputSeqLen,
    ComposableLayers numAttnLayers numEmbeds numHeads ffnDim inputSeqLen batchSize dtype device,
    TensorOptions '[inputSeqLen, inputSeqLen] dtype device,
    UT.TensorLike (ComputeHaskellType dtype),
    Fractional (ComputeHaskellType dtype),
    BasicArithmeticDTypeIsValid device dtype,
    ComparisonDTypeIsValid device dtype,
    ComparisonDTypeIsValid device 'D.Int64,
    KnownDType dtype,
    KnownDevice device
  ) =>
  GPT2 numAttnLayers numHeads ffnDim paddingIdx maxSeqLen vocabSize numEmbeds dtype device ->
  Tensor device 'D.Int64 '[batchSize, inputSeqLen] ->
  IO (Tensor device dtype '[batchSize, inputSeqLen, vocabSize])
transformerLM GPT2 {..} xTokens = do
  let positions =
        expand @'[batchSize, inputSeqLen, numEmbeds] True
          . embed tPosEmbedding
          . Torch.Typed.Tensor.toDType @D.Int64
          . linspace @inputSeqLen (0 :: Int)
          $ natValI @(inputSeqLen - 1)
      x = embed tEmbedding xTokens `add` positions
      blocks =
        hfoldr
          (ComposeLayers @batchSize @inputSeqLen @dtype @device (Just causalMask))
          (id :: Net (Tensor device dtype '[batchSize, inputSeqLen, numEmbeds]) (Tensor device dtype '[batchSize, inputSeqLen, numEmbeds]))
          tLayers
  y <- runNet blocks x
  return
    . forward tProj
    $ forward tFinalLN y

instance
  ( All KnownNat '[paddingIdx, numEmbeds, inputSeqLen, batchSize, inputSeqLen],
    IsSuffixOf '[numEmbeds] '[batchSize, inputSeqLen, numEmbeds],
    paddingIdx + 1 <= vocabSize,
    1 <= inputSeqLen,
    ComposableLayers numAttnLayers numEmbeds numHeads ffnDim inputSeqLen batchSize dtype device,
    TensorOptions '[inputSeqLen, inputSeqLen] dtype device,
    UT.TensorLike (ComputeHaskellType dtype),
    Fractional (ComputeHaskellType dtype),
    BasicArithmeticDTypeIsValid device dtype,
    ComparisonDTypeIsValid device dtype,
    ComparisonDTypeIsValid device 'D.Int64,
    KnownDType dtype,
    KnownDevice device
  ) =>
  HasForward (GPT2 numAttnLayers numHeads ffnDim paddingIdx inputSeqLen vocabSize numEmbeds dtype device) (Tensor device 'D.Int64 '[batchSize, inputSeqLen]) (Tensor device dtype '[batchSize, inputSeqLen, vocabSize])
  where
  forward model input = unsafePerformIO $ transformerLM model input
  forwardStoch model input = transformerLM model input

sinusoidal ::
  forall vocabSize numEmbeds device.
  ( All KnownNat '[vocabSize, numEmbeds],
    1 <= vocabSize,
    1 <= Div numEmbeds 2,
    (Div numEmbeds 2 * 2) ~ numEmbeds,
    StandardFloatingPointDTypeValidation device 'D.Float,
    BasicArithmeticDTypeIsValid device 'D.Float,
    KnownDevice device
  ) =>
  Tensor device 'D.Float '[vocabSize, numEmbeds]
sinusoidal =
  let positions =
        unsqueeze @1
          . linspace @vocabSize (0 :: Int)
          $ natValI @(vocabSize - 1)
      scalingFactors =
        exp
          . mulScalar (-log (10000 :: Double) / (fromInteger . natVal $ Proxy @(Div numEmbeds 2)))
          . linspace @(Div numEmbeds 2) (0 :: Int)
          $ natValI @((Div numEmbeds 2) - 1)
      radians = mul positions scalingFactors
      weights = stack @2 (sin radians :. cos radians :. HNil)
   in reshape weights

instance
  ( paddingIdx <= vocabSize,
    1 <= maxSeqLen,
    1 <= vocabSize - paddingIdx,
    1 <= Div numEmbeds 2,
    (((vocabSize - paddingIdx) - 1) + (1 + paddingIdx)) ~ vocabSize,
    (Div numEmbeds 2 * 2) ~ numEmbeds,
    All KnownNat '[ffnDim, paddingIdx, vocabSize, maxSeqLen, numEmbeds],
    HReplicate numAttnLayers (TransformerLayerSpec numEmbeds numHeads ffnDim dtype device),
    A.Randomizable
      (HList (HReplicateR numAttnLayers (TransformerLayerSpec numEmbeds numHeads ffnDim dtype device)))
      (HList (TransformerLayers numAttnLayers numEmbeds numHeads ffnDim dtype device)),
    KnownDType dtype,
    RandDTypeIsValid device dtype,
    StandardFloatingPointDTypeValidation device 'D.Float,
    BasicArithmeticDTypeIsValid device 'D.Float,
    KnownDevice device
  ) =>
  A.Randomizable
    (GPT2Spec numAttnLayers numHeads ffnDim paddingIdx maxSeqLen vocabSize numEmbeds dtype device)
    (GPT2 numAttnLayers numHeads ffnDim paddingIdx maxSeqLen vocabSize numEmbeds dtype device)
  where
  sample GPT2Spec {..} =
    GPT2
      <$> A.sample (LearnedEmbeddingWithRandomInitSpec @('Just paddingIdx))
      <*> A.sample (ConstEmbeddingSpec @'Nothing (Torch.Typed.Tensor.toDType sinusoidal))
      <*> A.sample (hreplicate @numAttnLayers lmLayerSpec)
      <*> A.sample (LayerNormSpec epsSpec'')
      <*> A.sample LinearSpec
