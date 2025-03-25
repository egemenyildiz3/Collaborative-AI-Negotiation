import logging
import time
from random import randint
from math import floor

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
        self.opponent_model: OpponentModel = None
        self.all_bids: AllBidsList = None
        self.sorted_bids = []  # Used to store bids sorted by our utility
        self.min_acceptable_util = 0.85  # This is a configurable minimum utility threshold

        self.logger.log(logging.INFO, "Group30Agent initialized")

    def notifyChange(self, data: Inform):
        if isinstance(data, Settings):
            # Called once at the beginning of the negotiation session
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

            # Pre-calculate all bids and sort them by our utility for efficient access
            self.all_bids = AllBidsList(self.domain)
            self.sorted_bids = sorted(
                [(bid, float(self.profile.getUtility(bid))) for bid in self.all_bids],
                key=lambda x: x[1], reverse=True
            )

        elif isinstance(data, ActionDone):
            action = data.getAction()
            actor = action.getActor()

            # Ignore our own actions
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
        return "A utility-focused agent that balances self-interest and opponent modeling."

    def opponent_action(self, action):
        if isinstance(action, Offer):
            if self.opponent_model is None:
                self.opponent_model = OpponentModel(self.domain)

            bid = action.getBid()
            self.opponent_model.update(bid)
            self.last_received_bid = bid

    def my_turn(self):
        # Decide whether to accept or offer a counter bid
        if self.accept_condition(self.last_received_bid):
            self.send_action(Accept(self.me, self.last_received_bid))
        else:
            offer_bid = self.find_bid()
            self.send_action(Offer(self.me, offer_bid))

    def save_data(self):
        # Save data for learning between sessions
        data = "Placeholder for persistent learning data."
        with open(f"{self.storage_dir}/data.md", "w") as f:
            f.write(data)

    def accept_condition(self, bid: Bid) -> bool:
        if bid is None:
            return False

        # Check time pressure using the progress object
        progress = self.progress.get(time.time() * 1000)

        # Accept if utility is above our target AND we are late in negotiation
        if self.profile.getUtility(bid) >= self.min_acceptable_util and progress > 0.97:
            return True

        # Accept if opponent’s offer is better than what we’re about to propose
        candidate = self.find_bid()
        return self.profile.getUtility(bid) >= self.profile.getUtility(candidate)

    def find_bid(self) -> Bid:
        # Take a top X% of the sorted bids and pick randomly within
        top_x_percent = 0.1  # We only consider top 10% of bids by our utility
        top_n = max(1, floor(top_x_percent * len(self.sorted_bids)))

        candidates = self.sorted_bids[:top_n]

        best_bid = None
        best_score = -1

        for bid, _ in candidates:
            score = self.score_bid(bid)
            if score > best_score:
                best_score = score
                best_bid = bid

        return best_bid

    def score_bid(self, bid: Bid, alpha: float = 0.9, eps: float = 0.1) -> float:
        # Combines our utility and predicted opponent utility based on time pressure
        progress = self.progress.get(time.time() * 1000)

        our_utility = float(self.profile.getUtility(bid))
        time_pressure = 1.0 - progress ** (1 / eps)
        score = alpha * time_pressure * our_utility

        if self.opponent_model is not None:
            opponent_utility = self.opponent_model.get_predicted_utility(bid)
            score += (1.0 - alpha * time_pressure) * opponent_utility

        return score
