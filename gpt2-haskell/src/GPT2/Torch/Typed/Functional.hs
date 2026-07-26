{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoStarIsType #-}

-- | Typed counterparts of ATen operations that "Torch.Typed.Functional" does
-- not cover, each with the shape relation it satisfies written into its type.
module GPT2.Torch.Typed.Functional
  ( geluApproximate,
    Multinomial,
    MultinomialCheck,
    multinomial,
  )
where

import GHC.TypeLits
import System.IO.Unsafe (unsafePerformIO)
import Torch.Internal.Cast (cast2)
import qualified Torch.Internal.Managed.Native as ATen.Managed
import Torch.Typed

-- | gelu with a selectable approximation.
--
-- 'Torch.Typed.Functional.gelu' always uses the exact erf formulation, while
-- GPT-2 was trained against the tanh approximation, which the ATen operator
-- selects through its @approximate@ argument.
geluApproximate ::
  forall shape dtype device.
  (GeluDTypeIsValid device dtype) =>
  -- | input
  Tensor device dtype shape ->
  -- | approximation, @"none"@ or @"tanh"@
  String ->
  -- | output
  Tensor device dtype shape
geluApproximate self approximate =
  unsafePerformIO $ cast2 ATen.Managed.gelu_ts self approximate

type family MultinomialCheck (n :: Nat) (shape :: [Nat]) (dim :: Nat) (sat :: Maybe Nat) (result :: Maybe a) :: a where
  MultinomialCheck _ shape dim _ 'Nothing = DimOutOfBound shape dim
  MultinomialCheck _ shape dim 'Nothing _ = DimOutOfBound shape dim
  MultinomialCheck n shape dim ('Just v) ('Just result) =
    If
      (n <=? v)
      result
      (TypeError (Text "n must be less than or equal to the number of elements in the first dim."))

-- | The shape of @'multinomial' \@n@ over @dim@: that dimension becomes the
-- number of samples drawn.
type Multinomial n shape dim = MultinomialCheck n shape dim (ExtractDim dim shape) (ReplaceDim dim shape n)

-- | Draw @samples@ indices from the categorical distribution each row of the
-- input describes.
multinomial ::
  forall samples dim shape shape' device dtype.
  ( KnownNat samples,
    KnownNat dim,
    StandardFloatingPointDTypeValidation device dtype,
    shape' ~ Multinomial samples shape dim,
    KnownDevice device
  ) =>
  Tensor device dtype shape ->
  IO (Tensor device 'Int64 shape')
multinomial input = cast2 ATen.Managed.multinomial_tl input (natValI @samples)
