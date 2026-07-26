{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# LANGUAGE NoStarIsType #-}

module Main where

import Control.Monad.Trans.Class (lift)
import Control.Monad.Trans.Maybe
import Data.ByteString.Char8 (pack)
import Data.Proxy
import GHC.Int (Int64)
import GHC.TypeLits
import GPT2.Loader
import GPT2.Model (transformerLM)
import GPT2.Torch.Typed.Functional (multinomial)
import SafeTensors hiding (shape)
import System.Environment (getArgs)
import Tiktoken (fromRanks, r50k_base, toRanks)
import qualified Torch as UT
import qualified Torch.DType as D
import Torch.Typed hiding (length, sample, toInt, transformerLM)
import Torch.Typed.Index (IndexType (..), getSlice)

-- | Sample one token from the distribution the model predicts for the last
-- position, restricted to its ten most likely continuations.
--
-- The context is described by the index of its last position, so the slice
-- @tensor[:, lastIdx]@ is in bounds by construction and 'getSlice' computes
-- the result shape at compile time.
sample ::
  forall lastIdx batchSize device dtype.
  ( All KnownNat [batchSize, lastIdx],
    dtype ~ D.Float,
    StandardFloatingPointDTypeValidation device dtype,
    KnownDevice device
  ) =>
  Tensor device dtype '[batchSize, lastIdx + 1, VocabSize] ->
  IO (Tensor device D.Int64 '[batchSize, 1])
sample tensor' = do
  ix <- multinomial @1 @1 topk_probs
  return $ gatherDim @1 ix topk_indices
  where
    logits = getSlice @'[ 'SliceAll, 'SliceAt lastIdx] tensor'
    (topk_probs, topk_indices) = topk @10 @1 True True $ softmax @1 logits

infer ::
  forall lastIdx.
  (KnownNat lastIdx) =>
  Model ->
  [[Int64]] ->
  IO (Tensor ModelDevice UT.Float '[1, lastIdx + 1, VocabSize])
infer model tokens =
  transformerLM model
    $ UnsafeMkTensor
      @ModelDevice
      @D.Int64
      @'[1, lastIdx + 1]
    $ UT.asTensor tokens

-- Extract the token from the tensor result
toInt :: Tensor device D.Int64 '[batchSize, 1] -> Int
toInt tensor = UT.asValue $ UT.toDType D.Int64 $ toDynamic tensor

-- Generate tokens autoregressively
generate :: Model -> [Int] -> Int -> MaybeT IO [Int]
generate _ tokens 0 = return tokens
generate _ [] _ = hoistMaybe Nothing
generate model tokens n =
  withNat (length tokens - 1) $ \(Proxy :: Proxy lastIdx) -> do
    logits <- lift $ infer @lastIdx model [map fromIntegral tokens]
    result <- lift $ sample @lastIdx logits
    let newToken = toInt result
    generate model (tokens ++ [newToken]) (n - 1)

runInference :: [String] -> MaybeT IO ()
runInference [] = lift $ putStrLn "No arguments provided"
runInference [fp :: FilePath] = do
  st <- lift $ readSafeTensors fp
  model <- loadGPT2FromSafeTensors st
  let inputText = "Hello, I'm a language model,"
  tokens <- hoistMaybe $ toRanks r50k_base (pack inputText)
  lift $ putStrLn $ "Input: " ++ inputText

  -- Generate 10 tokens autoregressively
  generatedTokens <- generate model tokens 10

  -- Convert tokens back to text and display
  lift $ putStrLn "Generated text:"
  lift $ print $ fromRanks r50k_base generatedTokens
runInference (fp : _) = runInference [fp]

main :: IO ()
main = do
  args <- getArgs
  _ <- runMaybeT $ runInference args
  return ()
