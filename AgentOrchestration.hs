{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE FlexibleContexts #-}

-- | Agent Orchestration as Kleisli Composition
--
-- A direct renaming of the categorical-evolution framework
-- (~/git/categorical-evolution/haskell/) to agent orchestration vocabulary.
--
-- The structural isomorphism:
--
--   Evolution.Category   ≅  Agent.Category
--   Evolution.Effects    ≅  Agent.Harness
--   Evolution.Operators  ≅  Agent.Tools
--   Evolution.Pipeline   ≅  Agent.Task
--   Evolution.Strategy   ≅  Agent.Workflow
--   Evolution.Island     ≅  Agent.Orchestrator
--   Evolution.Landscape  ≅  Agent.TaskAnalysis
--
-- Every theorem in the ACT 2026 paper transfers verbatim.

module AgentOrchestration where

import Control.Monad ((>=>))
import Control.Monad.Reader
import Control.Monad.State.Strict
import Control.Monad.Writer.Strict
import Data.List (sortBy)
import Data.Ord (comparing, Down(..))
import System.Random
import Prelude hiding (id)

-- ============================================================
-- Agent.Harness  (was: Evolution.Effects)
-- The monad capturing agent execution effects.
-- ============================================================

-- | Agent configuration. Read-only during execution.
-- Was: GAConfig
data HarnessConfig = HarnessConfig
  { poolSize       :: !Int      -- ^ Number of candidate solutions to maintain
  , perturbRate    :: !Double   -- ^ Probability of modifying each component
  , combineRate    :: !Double   -- ^ Probability of merging two solutions
  , selectPressure :: !Int      -- ^ How many candidates to compare when choosing
  , eliteKeep      :: !Int      -- ^ Number of top solutions preserved each round
  , maxRounds      :: !Int      -- ^ Maximum number of execution rounds
  } deriving (Show, Eq)

-- | Statistics for a single execution round.
-- Was: GenStats
data RoundStats = RoundStats
  { roundNumber   :: !Int
  , bestQuality   :: !Double
  , avgQuality    :: !Double
  , worstQuality  :: !Double
  , solutionDiversity :: !Double  -- ^ Diversity of solutions (0-1)
  } deriving (Show, Eq)

-- | Accumulated execution log. A monoid: we accumulate round stats.
-- Was: GALog
newtype HarnessLog = HarnessLog { rounds :: [RoundStats] }
  deriving (Show, Eq, Semigroup, Monoid)

-- | The harness monad: Reader for config, State for PRNG, Writer for logging.
--
-- Was: EvoM
--
-- This is the concrete monad that tool calls execute in.
-- Compare with the Claude Code harness analysis:
--   T = Reader(Config) × Writer(Log) × State(PRNG)
--
-- The Kleisli category for HarnessM is the category of agent operations.
newtype HarnessM a = HarnessM
  { unHarnessM :: ReaderT HarnessConfig (StateT StdGen (Writer HarnessLog)) a
  } deriving (Functor, Applicative, Monad,
              MonadReader HarnessConfig,
              MonadState StdGen,
              MonadWriter HarnessLog)

runHarnessM :: HarnessConfig -> StdGen -> HarnessM a -> (a, StdGen, HarnessLog)
runHarnessM config gen (HarnessM m) =
  let ((a, gen'), log') = runWriter (runStateT (runReaderT m config) gen)
  in (a, gen', log')

-- ============================================================
-- Agent.Category  (was: Evolution.Category)
-- The core categorical structure.
-- ============================================================

-- | An evaluated solution: a solution paired with its quality score.
-- Was: Scored
data Evaluated a = Evaluated
  { solution :: !a
  , quality  :: !Double
  } deriving (Show, Eq)

-- | A pool of candidate solutions.
-- Was: Population
type Pool a = [a]

-- | An agent operation: a morphism in the Kleisli category for HarnessM.
--
-- Was: GeneticOp
--
-- This is THE central type. AgentOps compose via (>>>:) to form tasks.
-- Each tool call in Claude Code is an AgentOp.
-- A complete task (pipeline of tool calls) is their Kleisli composite.
newtype AgentOp m a b = AgentOp { runOp :: [a] -> m [b] }

-- | Identity: pass through unchanged.
-- Was: idOp
idOp :: Monad m => AgentOp m a a
idOp = AgentOp return

-- | Left-to-right Kleisli composition.
-- Was: (>>>:)
--
-- This is the >>= of the harness monad, lifted to agent operations.
-- In Claude Code terms: chaining tool calls.
--
--   readFile >>>: parseResult >>>: editFile >>>: runTests
(>>>:) :: Monad m => AgentOp m a b -> AgentOp m b c -> AgentOp m a c
(AgentOp f) >>>: (AgentOp g) = AgentOp (f >=> g)
infixl 1 >>>:

-- | Lift a pure transformation.
liftPure :: Monad m => ([a] -> [b]) -> AgentOp m a b
liftPure f = AgentOp (return . f)

-- | Lift a monadic transformation.
liftMonadic :: ([a] -> m [b]) -> AgentOp m a b
liftMonadic = AgentOp

-- | Apply a pure function to each solution independently.
pointwise :: Monad m => (a -> b) -> AgentOp m a b
pointwise f = liftPure (map f)

-- | Apply a monadic function to each solution independently.
pointwiseM :: Monad m => (a -> m b) -> AgentOp m a b
pointwiseM f = AgentOp (mapM f)

-- ============================================================
-- Agent.Tools  (was: Evolution.Operators)
-- Concrete agent operations (tool calls).
-- ============================================================

-- | Assess: evaluate each solution against a quality function.
-- Was: evaluate
--
-- In agent terms: run tests, check quality, score the output.
assess :: (a -> Double) -> AgentOp HarnessM a (Evaluated a)
assess f = pointwise (\x -> Evaluated x (f x))

-- | Rank and select: keep the best solutions.
-- Was: tournamentSelect / elitistSelect
--
-- In agent terms: from N candidate outputs, pick the best ones.
rankSelect :: AgentOp HarnessM (Evaluated a) (Evaluated a)
rankSelect = liftMonadic $ \pool -> do
  cfg <- ask
  let n = poolSize cfg
      k = selectPressure cfg
  replicateM n (tournament k pool)
  where
    tournament k pool = do
      contestants <- replicateM k (randomChoice pool)
      return $ head $ sortBy (comparing (Down . quality)) contestants

-- | Combine: merge information from pairs of solutions.
-- Was: onePointCrossover
--
-- In agent terms: take the best parts of two different approaches
-- and combine them into a new approach.
combine :: AgentOp HarnessM (Evaluated [a]) (Evaluated [a])
combine = liftMonadic $ \pool -> do
  cfg <- ask
  let rate = combineRate cfg
  go rate pool
  where
    go _ [] = return []
    go _ [x] = return [x]
    go rate (p1:p2:rest) = do
      r <- randomDouble'
      children <- if r < rate
        then do
          let g1 = solution p1; g2 = solution p2
              len = min (length g1) (length g2)
          point <- randomInt' 1 (max 1 (len - 1))
          let (h1, t1) = splitAt point g1
              (h2, t2) = splitAt point g2
          return [Evaluated (h1 ++ t2) 0, Evaluated (h2 ++ t1) 0]
        else return [p1 { quality = 0 }, p2 { quality = 0 }]
      rest' <- go rate rest
      return (children ++ rest')

-- | Perturb: randomly modify parts of each solution.
-- Was: pointMutate
--
-- In agent terms: ask the LLM to vary/refactor a specific part.
perturb :: (a -> HarnessM a) -> AgentOp HarnessM (Evaluated [a]) (Evaluated [a])
perturb perturbFn = pointwiseM $ \scored -> do
  cfg <- ask
  let rate = perturbRate cfg
  components' <- mapM (\component -> do
      r <- randomDouble'
      if r < rate then perturbFn component else return component
    ) (solution scored)
  return (scored { solution = components', quality = 0 })

-- | Log round statistics.
-- Was: logGeneration
logRound :: Int -> AgentOp HarnessM (Evaluated a) (Evaluated a)
logRound rnd = liftMonadic $ \pool -> do
  let qualities = map quality pool
      best = maximum qualities
      worst = minimum qualities
      avg = sum qualities / fromIntegral (length qualities)
      variance = sum (map (\q -> (q - avg)^(2::Int)) qualities)
                 / fromIntegral (length qualities)
      div' = if best > worst then sqrt variance / (best - worst) else 0
  tell $ HarnessLog [RoundStats rnd best avg worst div']
  return pool

-- ============================================================
-- Agent.Task  (was: Evolution.Pipeline)
-- Composable task pipelines.
-- ============================================================

-- | A single round of agent execution.
-- Was: generationStep
--
-- assess >>>: logRound >>>: rankSelect >>>: combine >>>: perturb
--
-- This IS a Claude Code turn: evaluate the current state, log it,
-- select the best approach, combine insights, vary the solution.
roundStep :: ([a] -> Double) -> (a -> HarnessM a) -> Int
          -> AgentOp HarnessM [a] [a]
roundStep qualityFn perturbFn rnd =
  assess qualityFn
    >>>: logRound rnd
    >>>: rankSelect
    >>>: combine
    >>>: perturb perturbFn
    >>>: pointwise solution

-- | Result of running an agent task.
-- Was: EvoResult
data TaskResult a = TaskResult
  { bestSolution  :: !a
  , bestQuality'  :: !Double
  , finalPool     :: ![Evaluated a]
  , taskLog       :: !HarnessLog
  , finalPRNG     :: !StdGen
  } deriving (Show)

-- | Execute a task for N rounds.
-- Was: evolve
orchestrate :: forall a. ([a] -> Double) -> (a -> HarnessM a)
            -> HarnessConfig -> StdGen -> [[a]] -> TaskResult [a]
orchestrate qualityFn perturbFn config gen0 initPool =
  let maxR = maxRounds config
      (result, gen', log') = runHarnessM config gen0 $ go 0 maxR initPool
      scored = sortBy (comparing (Down . quality))
               (map (\s -> Evaluated s (qualityFn s)) result)
      best = head scored
  in TaskResult (solution best) (quality best) scored log' gen'
  where
    go :: Int -> Int -> [[a]] -> HarnessM [[a]]
    go rnd maxR pool
      | rnd >= maxR = return pool
      | otherwise = do
          pool' <- runOp (roundStep qualityFn perturbFn rnd) pool
          go (rnd + 1) maxR pool'

-- ============================================================
-- Agent.Workflow  (was: Evolution.Strategy)
-- Composable multi-phase workflows.
-- ============================================================

-- | Termination condition. Forms a Boolean algebra.
-- Was: StopWhen
data StopWhen
  = AfterRounds Int
  | QualityAbove Double
  | Plateau Int                 -- ^ No improvement for N rounds
  | StopOr StopWhen StopWhen
  | StopAnd StopWhen StopWhen
  deriving (Show, Eq)

-- | A workflow: a composable algorithm transforming evaluated solution pools.
-- Was: Strategy
--
-- Level 2 of the hierarchy: workflows compose tasks.
-- In Claude Code: plan → implement → test → refine
newtype Workflow a = Workflow
  { runWorkflow :: [Evaluated a] -> HarnessM (WorkflowResult a)
  }

data WorkflowResult a = WorkflowResult
  { resultPool  :: [Evaluated a]
  , resultBest  :: Evaluated a
  , resultRounds :: Int
  } deriving (Show)

-- | Sequential composition: run workflow A, then workflow B on A's output.
-- Was: sequential
--
-- This is composition in the workflow category.
-- plan >>>: implement = a workflow that plans then implements.
sequentialW :: Workflow a -> Workflow a -> Workflow a
sequentialW w1 w2 = Workflow $ \pool -> do
  r1 <- runWorkflow w1 pool
  r2 <- runWorkflow w2 (resultPool r1)
  let best = if quality (resultBest r1) >= quality (resultBest r2)
             then resultBest r1 else resultBest r2
  return $ WorkflowResult (resultPool r2) best (resultRounds r1 + resultRounds r2)

-- | Race: run both workflows independently, return the better result.
-- Was: race
--
-- In agent terms: spawn two sub-agents with different approaches,
-- keep whichever produces the better output.
raceW :: Workflow a -> Workflow a -> Workflow a
raceW w1 w2 = Workflow $ \pool -> do
  g <- get
  let (g1, g2) = split g
  put g1; r1 <- runWorkflow w1 pool
  put g2; r2 <- runWorkflow w2 pool
  return $ if quality (resultBest r1) >= quality (resultBest r2) then r1 else r2

-- | Adaptive: run primary, check predicate, fallback if needed.
-- Was: adaptive
--
-- In agent terms: try approach A; if it's not working (plateau detection,
-- quality threshold), switch to approach B. This is the three-strike rule.
adaptiveW :: (WorkflowResult a -> Bool) -> Workflow a -> Workflow a -> Workflow a
adaptiveW predicate primary fallback = Workflow $ \pool -> do
  r1 <- runWorkflow primary pool
  if predicate r1 then return r1
  else do
    r2 <- runWorkflow fallback (resultPool r1)
    let best = if quality (resultBest r1) >= quality (resultBest r2)
               then resultBest r1 else resultBest r2
    return $ WorkflowResult (resultPool r2) best (resultRounds r1 + resultRounds r2)

-- ============================================================
-- Agent.Orchestrator  (was: Evolution.Island / Evolution.Strategy.islandStrategy)
-- The island functor: lifts workflows to multi-agent systems.
-- ============================================================

-- | Migration topology between sub-agents.
-- Was: Topology / IslandTopo
--
-- THE key parameter from the paper. Determines diversity dynamics
-- independently of the task domain.
data SyncTopology
  = NoSync           -- ^ Strict: complete isolation. Maximum diversity.
  | RingSync         -- ^ Each agent shares with one neighbor. Moderate diversity.
  | StarSync         -- ^ One coordinator, isolated workers. Moderate-low diversity.
  | FullSync         -- ^ Every agent shares with every other. Minimum diversity.
  deriving (Show, Eq)

-- | Configuration for the multi-agent orchestrator.
-- Was: IslandModel / IslandConfig
data OrchestratorConfig = OrchestratorConfig
  { numAgents     :: !Int            -- ^ Number of sub-agents
  , syncRate      :: !Double         -- ^ Fraction of solutions to share
  , syncFrequency :: !Int            -- ^ Share every N rounds
  , syncTopology  :: !SyncTopology   -- ^ How agents are connected
  } deriving (Show, Eq)

-- | The orchestrator functor I: lifts a step function to multi-agent.
-- Was: islandStrategy
--
-- Level 3 of the hierarchy.
--
-- == The Strict/Lax Dichotomy (Theorem from the paper)
--
-- Let I(μ, freq, topo) be the orchestrator functor.
--
-- (i) Strict case: If syncRate = 0 or syncFrequency > maxRounds,
--     then I is a strict 2-functor:
--       I(W₁ ; W₂) ≡ I(W₁) ; I(W₂)
--     Sub-agent workflows compose cleanly. No information leaks.
--
-- (ii) Lax case: If syncRate > 0 and syncFrequency ≤ maxRounds,
--      then I is lax. The laxator norm (error from treating I as strict)
--      depends on syncRate and syncFrequency.
--
-- Practical consequence: you cannot reason about composed multi-agent
-- workflows by reasoning about agents independently, regardless of
-- how rare the syncing is. One sync event changes the category.
--
-- This is why "dangerously skip all permissions" in Claude Code
-- collapses the interaction tree to a Kleisli category — you've
-- removed all sync points (user approval) from the topology.
orchestrateMulti :: OrchestratorConfig
                 -> (Int -> [Evaluated a] -> HarnessM [Evaluated a])
                 -> StopWhen
                 -> Workflow a
orchestrateMulti config step stop = Workflow $ \pool0 -> do
  let n = numAgents config
      agentPools = splitInto n pool0
      merged = bestOf (concat agentPools)
  (finalPools, finalBest, totalRounds) <- agentLoop 0 merged agentPools
  return $ WorkflowResult (concat finalPools) finalBest totalRounds
  where
    agentLoop rnd overallBest pools
      | shouldStop stop rnd overallBest = return (pools, overallBest, rnd)
      | otherwise = do
          -- Each agent executes one round independently
          pools' <- mapM (step rnd) pools
          -- Sync according to topology
          pools'' <- if rnd > 0 && rnd `mod` syncFrequency config == 0
                     then syncAgents config pools'
                     else return pools'
          let merged = concat pools''
              currentBest = bestOf merged
              overallBest' = if quality currentBest >= quality overallBest
                             then currentBest else overallBest
          agentLoop (rnd + 1) overallBest' pools''

-- | Sync (migrate) solutions between agents according to topology.
-- Was: migrateIslands
syncAgents :: OrchestratorConfig -> [[Evaluated a]] -> HarnessM [[Evaluated a]]
syncAgents config pools = case syncTopology config of
  NoSync   -> return pools  -- Strict: no communication
  RingSync -> ringSync (syncRate config) pools
  StarSync -> starSync (syncRate config) pools
  FullSync -> fullSync (syncRate config) pools

-- Ring: agent i shares with agent (i+1) mod n
ringSync :: Double -> [[Evaluated a]] -> HarnessM [[Evaluated a]]
ringSync rate pools
  | length pools <= 1 = return pools
  | otherwise = do
      let migCounts = map (\p -> max 1 (round (rate * fromIntegral (length p)) :: Int)) pools
      migrants <- mapM (\(p, k) -> do
          shuffled <- shuffle' p
          return (take k shuffled)
        ) (zip pools migCounts)
      let updated = zipWith (\p migIn ->
            let trimmed = take (length p - length migIn) p
            in migIn ++ trimmed
            ) pools (last migrants : init migrants)
      return updated

-- Star: agent 0 is hub; all others receive from hub only
starSync :: Double -> [[Evaluated a]] -> HarnessM [[Evaluated a]]
starSync rate pools
  | length pools <= 1 = return pools
  | otherwise = do
      let hub = head pools
          k = max 1 (round (rate * fromIntegral (length hub)) :: Int)
      hubMigrants <- do
        shuffled <- shuffle' hub
        return (take k shuffled)
      let updated = hub : map (\p ->
            let trimmed = take (length p - length hubMigrants) p
            in hubMigrants ++ trimmed
            ) (tail pools)
      return updated

-- Fully connected: every agent shares with every other
fullSync :: Double -> [[Evaluated a]] -> HarnessM [[Evaluated a]]
fullSync rate pools
  | length pools <= 1 = return pools
  | otherwise = do
      let n = length pools
          perAgent = max 1 (round (rate * fromIntegral (length (head pools))
                                   / fromIntegral (n - 1)) :: Int)
      donors <- mapM (\p -> do
          shuffled <- shuffle' p
          return (take (perAgent * (n - 1)) shuffled)
        ) pools
      let updated = zipWith3 (\i p _ ->
            let incoming = concatMap (\(j, d) ->
                  if j /= i then take perAgent d else [])
                  (zip [0 :: Int ..] donors)
                trimmed = take (length p - length incoming) p
            in incoming ++ trimmed
            ) [0 :: Int ..] pools donors
      return updated

-- ============================================================
-- Agent.TaskAnalysis  (was: Evolution.Landscape)
-- Analyze a task to recommend a workflow.
-- ============================================================

-- | Task profile: characterizes the "landscape" of a problem.
-- Was: LandscapeProfile
--
-- In agent terms: before choosing an orchestration strategy,
-- analyze the task to determine which approach fits.
data TaskProfile = TaskProfile
  { taskComplexity   :: !Double  -- ^ 0 = straightforward, 1 = highly complex
  , solutionVariance :: !Double  -- ^ How much do solutions vary?
  , feedbackClarity  :: !Double  -- ^ Is the quality signal clear or noisy?
  } deriving (Show, Eq)

-- | Recommend a workflow based on task analysis.
-- Was: recommendStrategy
--
-- This is a functor from task profiles to workflows:
--   analyzeLandscape >>> recommendStrategy
recommendWorkflow :: TaskProfile
                  -> ([a] -> Double) -> (a -> HarnessM a)
                  -> StopWhen -> Workflow a
recommendWorkflow profile qualityFn perturbFn stop
  | taskComplexity profile > 0.7 =
      -- Complex task: race two approaches (exploration + exploitation)
      raceW (standardWorkflow qualityFn perturbFn stop)
            (standardWorkflow qualityFn perturbFn stop)  -- (would use different params)
  | feedbackClarity profile < 0.3 =
      -- Noisy feedback: use steady exploration
      standardWorkflow qualityFn perturbFn stop
  | otherwise =
      -- Standard: generational approach
      standardWorkflow qualityFn perturbFn stop

-- Placeholder for a standard workflow
standardWorkflow :: ([a] -> Double) -> (a -> HarnessM a) -> StopWhen -> Workflow a
standardWorkflow _ _ _ = Workflow $ \pool ->
  return $ WorkflowResult pool (bestOf pool) 0

-- ============================================================
-- Utility functions (internal)
-- ============================================================

randomDouble' :: HarnessM Double
randomDouble' = state (uniformR (0.0, 1.0))

randomInt' :: Int -> Int -> HarnessM Int
randomInt' lo hi = state (uniformR (lo, hi))

randomChoice :: [a] -> HarnessM a
randomChoice [] = error "randomChoice: empty list"
randomChoice xs = do
  i <- randomInt' 0 (length xs - 1)
  return (xs !! i)

shuffle' :: [a] -> HarnessM [a]
shuffle' [] = return []
shuffle' [x] = return [x]
shuffle' xs = go (length xs - 1) xs
  where
    go 0 ys = return ys
    go i ys = do
      j <- randomInt' 0 i
      let ys' = swapAt i j ys
      go (i - 1) ys'
    swapAt i j ys
      | i == j    = ys
      | otherwise =
          [ if k == i then ys !! j
            else if k == j then ys !! i
            else y
          | (k, y) <- zip [0..] ys
          ]

bestOf :: [Evaluated a] -> Evaluated a
bestOf = head . sortBy (comparing (Down . quality))

shouldStop :: StopWhen -> Int -> Evaluated a -> Bool
shouldStop (AfterRounds n) rnd _ = rnd >= n
shouldStop (QualityAbove t) _ e  = quality e >= t
shouldStop (Plateau _) _ _       = False  -- Simplified
shouldStop (StopOr a b) r e      = shouldStop a r e || shouldStop b r e
shouldStop (StopAnd a b) r e     = shouldStop a r e && shouldStop b r e

splitInto :: Int -> [a] -> [[a]]
splitInto 0 xs = [xs]
splitInto 1 xs = [xs]
splitInto n xs =
  let size = length xs `div` n
      (chunk, rest) = splitAt size xs
  in chunk : splitInto (n - 1) rest
