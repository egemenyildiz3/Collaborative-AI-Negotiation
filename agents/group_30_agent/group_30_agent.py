import logging
import time
from random import randint
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
        self.last_offered_bid: Bid = None
        self.opponent_model: OpponentModel = None
        self.frequency_model: FrequencyOpponentModel = None
        self.all_bids: AllBidsList = None
        self.reservation_value = Decimal(0.4)
        self.received_bids = []
        self.social_welfare_bid = None
        self.social_welfare_value = Decimal(0.0)
        self.opponent_stubborn = False  # Detected late in negotiation if needed

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
            self.frequency_model = FrequencyOpponentModel.WithAction(self.frequency_model, action, self.progress.get(time.time() * 1000))

            self.last_received_bid = bid
            self.received_bids.append(bid)

            # Track best social welfare bid (as potential fallback)
            util = self.profile.getUtility(bid)
            if util > self.social_welfare_value:
                self.social_welfare_value = util
                self.social_welfare_bid = bid

    def my_turn(self):
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

        # If time is nearly up and opponent has shown stubbornness, accept fallback
        if progress > 0.98 and self.opponent_stubborn and self.profile.getUtility(bid) > self.social_welfare_value:
            return True

        return False

    def find_bid(self) -> Bid:
        bids = [self.all_bids.get(randint(0, self.all_bids.size() - 1)) for _ in range(5000)]

        progress = self.progress.get(time.time() * 1000)
        best_bid = None
        best_score = -1.0

        # Early stage: try to propose better and confident bids
        if progress < 0.3:
            if self.last_offered_bid is None:
                # Try to aim above a starting reservation value
                target = Decimal(0.75)
                bids.sort(key=lambda b: abs(self.profile.getUtility(b) - target))
                best_bid = bids[0]
            else:
                # Find bids better than last offer
                bids.sort(key=lambda b: self.profile.getUtility(b))
                threshold = self.profile.getUtility(self.last_offered_bid)
                filtered = [b for b in bids if self.profile.getUtility(b) >= threshold]
                best_bid = max(filtered, key=self.score_bid) if filtered else self.last_offered_bid

        elif progress < 0.98:
            # Middle phase: negotiate
            bids = [b for b in bids if self.profile.getUtility(b) >= self.profile.getUtility(self.last_offered_bid) * Decimal(0.9)]
            bids.sort(key=lambda b: -self.opponent_model.get_predicted_utility(b))
            best_bid = bids[0] if bids else self.last_offered_bid

        else:
            # Near end: fallback if opponent is unwilling
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