import logging
import time
from random import randint, choice
from math import floor
import copy
from decimal import Decimal

from geniusweb.actions.Accept import Accept
from geniusweb.actions.Action import Action
from geniusweb.actions.Offer import Offer
from geniusweb.actions.PartyId import PartyId
from geniusweb.bidspace.AllBidsList import AllBidsList
from geniusweb.inform.ActionDone import ActionDone
from geniusweb.inform.Finished import Finished
from geniusweb.inform.Inform import Inform
from geniusweb.inform.Settings import Settings
from geniusweb.inform.YourTurn import YourTurn
from geniusweb.issuevalue.Bid import Bid
from geniusweb.issuevalue.Domain import Domain
from geniusweb.party.Capabilities import Capabilities
from geniusweb.party.DefaultParty import DefaultParty
from geniusweb.profile.utilityspace.LinearAdditiveUtilitySpace import LinearAdditiveUtilitySpace
from geniusweb.profileconnection.ProfileConnectionFactory import ProfileConnectionFactory
from geniusweb.progress.ProgressTime import ProgressTime
from geniusweb.references.Parameters import Parameters
from tudelft_utilities_logging.ReportToLogger import ReportToLogger

from .utils.opponent_model import OpponentModel
from geniusweb.opponentmodel.FrequencyOpponentModel import FrequencyOpponentModel


class Group30Agent(DefaultParty):
    def __init__(self):
        super().__init__()
        self.logger: ReportToLogger = self.getReporter()

        self.domain: Domain = None
        self.parameters: Parameters = None
        self.profile: LinearAdditiveUtilitySpace = None
        self.progress: ProgressTime = None
        self.me: PartyId = None
        self.other: str = None
        self.settings: Settings = None
        self.storage_dir: str = None

        self.last_received_bid: Bid = None
        self.last_last_received_bid: Bid = None
        self.last_offered_bid: Bid = None

        # Variables for move size and thresholds (delta values)
        self.move_size = 0.0  # difference in utility between sequential bids
        self.silent_move_delta = 0.01
        self.concession_move_delta = 0.03  # moderate concession threshold
        self.large_concession_delta = 0.06   # large concession threshold
        self.selfish_move_delta = 0.01       # threshold for a selfish move

        # Opponent modeling and bid storage
        self.opponent_model: OpponentModel = None
        self.frequency_model: FrequencyOpponentModel = None
        self.all_bids: AllBidsList = None
        self.reservation_value = Decimal(0.4)
        self.received_bids = []  # overall opponent bid history
        self.social_welfare_bid = None
        self.social_welfare_value = Decimal(0.0)
        self.opponent_stubborn = False  # Detected late in negotiation if needed

        # New attributes for strategy recognition (used only after round 30)
        self.recognition_h_bid_history = []  # our bids during the recognition phase
        self.recognition_o_bid_history = []  # opponent bids during the recognition phase
        self.opponent_hypothesis = None        # e.g. {"HH", "R"} or {"CC", "R", "TT"}
        self.recognized_strategy = None        # final decision: "HH", "CC", "TT", or "R"

        self.logger.log(logging.INFO, "Group30Agent initialized")

    def notifyChange(self, data: Inform):
        if isinstance(data, Settings):
            self.settings = data
            self.me = self.settings.getID()
            self.progress = self.settings.getProgress()
            self.parameters = self.settings.getParameters()
            self.storage_dir = self.parameters.get("storage_dir")

            profile_connection = ProfileConnectionFactory.create(
                data.getProfile().getURI(), self.getReporter())
            self.profile = profile_connection.getProfile()
            self.domain = self.profile.getDomain()
            profile_connection.close()

            # Pre-load all bids once to reduce future overhead
            self.all_bids = AllBidsList(self.domain)

        elif isinstance(data, ActionDone):
            action = data.getAction()
            actor = action.getActor()
            if actor != self.me:
                self.other = str(actor).rsplit("_", 1)[0]
                self.opponent_action(action)

        elif isinstance(data, YourTurn):
            self.my_turn()

        elif isinstance(data, Finished):
            self.logger.log(logging.INFO, "Negotiation finished.")
            self.save_data()
            super().terminate()

        else:
            self.logger.log(logging.WARNING, "Unknown Inform received: " + str(data))

    def getCapabilities(self) -> Capabilities:
        return Capabilities(set(["SAOP"]), set(["geniusweb.profile.utilityspace.LinearAdditive"]))

    def send_action(self, action: Action):
        self.getConnection().send(action)

    def getDescription(self) -> str:
        return "Group30Agent"

    def opponent_action(self, action):
        if isinstance(action, Offer):
            if self.opponent_model is None:
                self.opponent_model = OpponentModel(self.domain)
                self.frequency_model = FrequencyOpponentModel.create().With(self.domain, None)

            bid = action.getBid()
            self.opponent_model.update(bid)
            self.frequency_model = FrequencyOpponentModel.WithAction(
                self.frequency_model, action, self.progress.get(time.time() * 1000)
            )

            # Update overall history
            if self.received_bids:
                self.last_last_received_bid = self.last_received_bid
                self.last_received_bid = bid
            else:
                self.last_last_received_bid = bid
                self.last_received_bid = bid
            self.received_bids.append(bid)

            # If we are in the recognition phase (after round 30 but before round 34), record in separate history.
            if 10 <= len(self.received_bids) <= 13:
                self.recognition_o_bid_history.append(bid)

            # Track best social welfare bid (as potential fallback)
            util = self.profile.getUtility(bid)
            if util > self.social_welfare_value:
                self.social_welfare_value = util
                self.social_welfare_bid = bid

            # Update our strategy recognition hypothesis (only during the recognition phase)
            if 10 <= len(self.received_bids) <= 13:
                self.update_strategy_recognition()

    def my_turn(self):
        # Use strategy recognition moves only during rounds 30-33.
        if 10 <= len(self.received_bids) <= 13:
            candidate = self.strategy_recognition_move()
            self.recognition_h_bid_history.append(candidate)
        else:
            candidate = self.find_bid()
        if self.accept_condition(self.last_received_bid, candidate):
            self.send_action(Accept(self.me, self.last_received_bid))
        else:
            self.last_offered_bid = candidate
            self.send_action(Offer(self.me, candidate))

    def save_data(self):
        with open(f"{self.storage_dir}/data.md", "w") as f:
            f.write("Learning data placeholder")

    def accept_condition(self, bid: Bid, next_bid: Bid) -> bool:
        if bid is None:
            return False

        progress = self.progress.get(time.time() * 1000)

        # Accept if opponent's offer is better than our next bid
        if self.score_bid(bid) > self.score_bid(next_bid):
            return True

        # Accept near deadline if it's a decent offer
        if self.profile.getUtility(bid) > 0.75 and progress > 0.95:
            return True

        # Accept fallback if opponent is stubborn near deadline
        if progress > 0.98 and self.opponent_stubborn and self.profile.getUtility(bid) > self.social_welfare_value:
            return True

        if self.profile.getUtility(bid) > 0.90:
            return True

        return False

    def find_bid(self) -> Bid:
        bids = [self.all_bids.get(randint(0, self.all_bids.size() - 1)) for _ in range(5000)]
        progress = self.progress.get(time.time() * 1000)
        best_bid = None
        best_score = -1.0

        if progress < 0.3:
            if self.last_offered_bid is None:
                target = Decimal(0.75)
                bids.sort(key=lambda b: abs(self.profile.getUtility(b) - target))
                best_bid = bids[0]
            else:
                bids.sort(key=lambda b: self.profile.getUtility(b))
                threshold = self.profile.getUtility(self.last_offered_bid)
                filtered = [b for b in bids if self.profile.getUtility(b) >= threshold]
                best_bid = max(filtered, key=self.score_bid) if filtered else self.last_offered_bid
        elif progress < 0.98:
            bids = [b for b in bids if self.profile.getUtility(b) >= self.profile.getUtility(self.last_offered_bid) * Decimal(0.9)]
            bids.sort(key=lambda b: -self.opponent_model.get_predicted_utility(b))
            best_bid = bids[0] if bids else self.last_offered_bid
        else:
            avg_opp_util = sum(Decimal(self.opponent_model.get_predicted_utility(b)) for b in self.received_bids) / len(self.received_bids)
            avg_my_util = sum(self.profile.getUtility(b) for b in self.received_bids) / len(self.received_bids)
            if avg_opp_util - avg_my_util > 0.4 or avg_my_util < self.reservation_value:
                self.opponent_stubborn = True
                return self.social_welfare_bid
            else:
                return self.last_offered_bid

        return best_bid

    def score_bid(self, bid: Bid, alpha: float = 0.9, eps: float = 0.1) -> float:
        progress = self.progress.get(time.time() * 1000)
        our_utility = float(self.profile.getUtility(bid))
        time_pressure = 1.0 - progress ** (1 / eps)
        score = alpha * time_pressure * our_utility

        if self.opponent_model is not None:
            opponent_utility = self.opponent_model.get_predicted_utility(bid)
            score += (1.0 - alpha * time_pressure) * opponent_utility

        return score

    def strategy_recognition_move(self) -> Bid:
        """
        Implements the four rounds for strategy recognition.
        This method is only invoked when len(received_bids) is between 30 and 33.
        We use the recognition histories (recognition_h_bid_history and recognition_o_bid_history)
        and treat the first recognition move as Round 1, second as Round 2, etc.
          - Round 1: Use a high-utility bid.
          - Round 2: Choose a bid that is a moderate concession.
          - Round 3: Depending on the hypothesis, choose either a large concession or a moderately selfish move.
          - Round 4: Choose a silent move and perform final analysis.
        """
        # Determine current recognition round based on our recognition history length.
        current_round = len(self.recognition_h_bid_history) + 1

        if current_round == 1:
            # Round 1: For example, choose our best bid.
            self.logger.log(logging.INFO, "Round 1 (Recognition): Selected best bid")
            bid = self.find_bid()
            return bid

        elif current_round == 2:
            # Round 2: Choose a bid that constitutes a moderate concession.
            bids = [self.all_bids.get(randint(0, self.all_bids.size() - 1)) for _ in range(5000)]
            candidates = [bid for bid in bids
                          if self.classify_self_move(self.recognition_h_bid_history[0], bid) == "concession"]
            if not candidates:
                candidates = [self.all_bids.get(randint(0, self.all_bids.size() - 1))]
            chosen = max(candidates, key=self.score_bid)
            self.logger.log(logging.INFO, "Round 2 (Recognition): Selected bid with moderate concession.")
            return chosen

        elif current_round == 3:
            # Round 3: Use our current hypothesis to select our move.
            if self.opponent_hypothesis is None:
                self.opponent_hypothesis = {"CC", "R", "TT"}
            if self.opponent_hypothesis == {"HH", "R"}:
                # Choose bid with a large concession.
                bids = [self.all_bids.get(randint(0, self.all_bids.size() - 1)) for _ in range(5000)]
                candidates = [bid for bid in bids
                              if self.classify_self_move(self.recognition_h_bid_history[-1], bid) == "large_concession"]
                self.logger.log(logging.INFO, "Round 3 (Recognition): (Hypothesis HH,R) Selected bid with large concession.")
            else:
                # Otherwise, choose a moderately selfish move.
                bids = [self.all_bids.get(randint(0, self.all_bids.size() - 1)) for _ in range(5000)]
                candidates = [bid for bid in bids
                              if self.classify_self_move(self.recognition_h_bid_history[-1], bid) == "selfish"]
                self.logger.log(logging.INFO, "Round 3 (Recognition): (Hypothesis CC,TT,R) Selected bid with moderate selfishness.")
            if not candidates:
                candidates = [self.all_bids.get(randint(0, self.all_bids.size() - 1))]
            chosen = max(candidates, key=self.score_bid)
            return chosen

        elif current_round == 4:
            # Round 4: Choose a silent move.
            candidates = [bid for bid in self.all_bids
                          if self.classify_self_move(self.recognition_h_bid_history[-1], bid) == "silent"]
            if not candidates:
                candidates = [self.all_bids.get(randint(0, self.all_bids.size() - 1))]
            chosen = max(candidates, key=self.score_bid)
            self.logger.log(logging.INFO, "Round 4 (Recognition): Selected silent move bid.")
            # Perform final analysis based on the recognition-phase opponent moves.
            self.final_strategy_analysis()
            return chosen

    def update_strategy_recognition(self):
        """
        During the recognition phase (rounds 1-4 in recognition history), update our hypothesis or finalize
        the recognized strategy using the opponent's moves recorded in recognition_o_bid_history.
        """
        # Analysis after Recognition Round 2:
        if len(self.recognition_o_bid_history) >= 2 and len(self.recognition_h_bid_history) >= 2 and self.opponent_hypothesis is None:
            sigma_opp = self.score_bid(self.recognition_o_bid_history[1]) - self.score_bid(self.recognition_o_bid_history[0])
            sigma_self = self.score_bid(self.recognition_h_bid_history[1]) - self.score_bid(self.recognition_h_bid_history[0])
            move_type_opp = self.classify_opponent_move(self.recognition_o_bid_history[0], self.recognition_o_bid_history[1])
            self.logger.log(logging.INFO, "Recognition: Opponent move is " + move_type_opp)
            self.logger.log(logging.INFO, "Recognition: Opponent sigma is " + str(sigma_opp))
            self.logger.log(logging.INFO, "Recognition: Self sigma is " + str(sigma_self))
            if sigma_opp <= abs(sigma_self) and move_type_opp in {"silent", "concession","small_concession","selfish"}:
                self.opponent_hypothesis = {"HH", "R"}
                self.logger.log(logging.INFO, "Recognition: Hypothesis updated to {HH, R}.")
            elif sigma_opp >= sigma_self:
                self.opponent_hypothesis = {"CC", "R", "TT"}
                self.logger.log(logging.INFO, "Recognition: Hypothesis updated to {CC, R, TT}.")

        # Analysis after Recognition Round 3:
        if len(self.recognition_o_bid_history) >= 3 and len(self.recognition_h_bid_history) >= 3 and self.recognized_strategy is None:
            sigma_opp = self.score_bid(self.recognition_o_bid_history[2]) - self.score_bid(self.recognition_o_bid_history[1])
            sigma_self = self.score_bid(self.recognition_h_bid_history[2]) - self.score_bid(self.recognition_h_bid_history[1])
            move_type_opp = self.classify_opponent_move(self.recognition_o_bid_history[1], self.recognition_o_bid_history[2])
            self.logger.log(logging.INFO, "Recognition: Opponent move is " + move_type_opp)
            self.logger.log(logging.INFO, "Recognition: Opponent sigma is " + str(sigma_opp))
            self.logger.log(logging.INFO, "Recognition: Self sigma is " + str(sigma_self))
            if self.opponent_hypothesis == {"HH", "R"}:
                if move_type_opp in {"selfish", "concession","small_concession","silent"} and abs(sigma_opp) <= abs(sigma_self):
                    self.recognized_strategy = "HH"
                    self.logger.log(logging.INFO, "Recognition: Concluded opponent plays Hardheaded (HH).")
                else:
                    self.recognized_strategy = "R"
                    self.logger.log(logging.INFO, "Recognition: Concluded opponent plays Random (R).")
            elif self.opponent_hypothesis == {"CC", "R", "TT"}:
                if move_type_opp == "selfish":
                    self.opponent_hypothesis = {"TT", "R"}
                    self.logger.log(logging.INFO, "Recognition: Updated hypothesis to {TT, R} (not Conceder).")
                elif move_type_opp in {"silent", "concession"}:
                    self.opponent_hypothesis = {"CC", "R"}
                    self.logger.log(logging.INFO, "Recognition: Updated hypothesis to {CC, R} (not Tit-for-Tat).")

    def classify_self_move(self, old_bid: Bid, new_bid: Bid) -> str:
        """
        Classify our move (from our perspective) based on the difference in scores.
        Returns:
          - "silent" if the change is very small.
          - "concession" if utility decreases moderately.
          - "large_concession" if utility decreases significantly.
          - "selfish" if utility increases beyond the threshold.
        """
        diff = self.score_bid(new_bid) - self.score_bid(old_bid)
        if abs(diff) <= self.silent_move_delta:
            return "silent"
        if diff < 0:
            if abs(diff) >= self.large_concession_delta:
                return "large_concession"
            elif abs(diff) >= self.concession_move_delta:
                return "concession"
            else:
                return "small_concession"
        elif diff > 0:
            if diff >= self.selfish_move_delta:
                return "selfish"
            else:
                return "small_selfish"
        return "unknown"

    def classify_opponent_move(self, old_bid: Bid, new_bid: Bid) -> str:
        """
        Classify the opponent's move from our perspective using our utility function.
        Using the definitions:
          - Silent move if |σ_H(μ)| ≤ δ.
          - Concession move (for opponent) if σ_H(μ) > δ.
          - Selfish move (for opponent) if σ_H(μ) < -δ.
        For small differences beyond δ, we return "small_concession" or "small_selfish" accordingly.
        """
        diff = self.opponent_model.get_predicted_utility(new_bid) - self.opponent_model.get_predicted_utility(old_bid)
        self.logger.log(logging.INFO, str(self.opponent_model.get_predicted_utility(new_bid)) + " " + str(self.opponent_model.get_predicted_utility(old_bid)) + " " + str(diff))
        if abs(diff) <= self.silent_move_delta:
            return "silent"
        elif diff > self.concession_move_delta:
            # A sufficiently positive diff: opponent's bid is better for us → concession.
            return "concession"
        elif diff < -self.selfish_move_delta:
            # A sufficiently negative diff: opponent's bid is worse for us → selfish move.
            return "selfish"
        else:
            # For differences that are not strong enough, we return a "small" variant.
            if diff > 0:
                return "small_concession"
            else:
                return "small_selfish"

    def final_strategy_analysis(self):
        """
        Final analysis after Recognition Round 4:
          - For Hypothesis {TT, R}: If the opponent’s move (from recognition round 2 to 3) is silent, conclude Tit-for-Tat (TT),
            otherwise classify as Random (R).
          - For Hypothesis {CC, R}: If the opponent’s move is silent or a concession, conclude Conceder (CC),
            otherwise classify as Random (R).
        """
        if len(self.recognition_o_bid_history) < 3:
            self.logger.log(logging.INFO, "Final Analysis: Not enough opponent data for analysis.")
            return
        move_type = self.classify_opponent_move(self.recognition_o_bid_history[1], self.recognition_o_bid_history[2])
        self.logger.log(logging.INFO, str(self.recognized_strategy))

        if self.recognized_strategy:
            self.logger.log(logging.INFO, "Final Analysis: Concluded opponent plays " + str(self.recognized_strategy) + ".")
            return
        if self.opponent_hypothesis == {"TT", "R"}:
            if move_type == "silent":
                self.recognized_strategy = "TT"
                self.logger.log(logging.INFO, "Final Analysis: Concluded opponent plays Tit-for-Tat (TT).")
            else:
                self.recognized_strategy = "R"
                self.logger.log(logging.INFO, "Final Analysis: Concluded opponent plays Random (R).")
        elif self.opponent_hypothesis == {"CC", "R"}:
            if move_type in {"silent", "concession"}:
                self.recognized_strategy = "CC"
                self.logger.log(logging.INFO, "Final Analysis: Concluded opponent plays Conceder (CC).")
            else:
                self.recognized_strategy = "R"
                self.logger.log(logging.INFO, "Final Analysis: Concluded opponent plays Random (R).")
        else:
            self.logger.log(logging.INFO, "Final Analysis: No clear hypothesis; defaulting to Random (R).")
            self.recognized_strategy = "R"
