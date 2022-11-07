import torch
import torch.optim as optimiser
from torch.distributions import Categorical
import torch.nn.functional as F


class PolicyGradient:
    """
    TODO: Change all mean to sum, so that it is consistent with Eccles.
    TODO: Introduce baseline, because it sucks and I think the issue might be how it is assigning value to states
    """

    def __init__(self, config):
        self.config = config
        self.policy = self.config['model']
        self.policy.to(self.config['device'])
        self.memory = dict.fromkeys(['log_prob', 'rewards', 'action_dist', 'states', 'action_dist_no_m',
                                     'values', 'log_prob_old_lang', 'actions'])
        self.config['optimiser'] = optimiser.Adam(self.policy.parameters(), lr=self.config['lr'])
        self.iterations = -1
        self.unique_agents = []
        self.move_cuda(self.config['device'])

    def move_cuda(self, device):
        """
        :param device:
        :return:
        """
        self.config['device'] = device
        self.policy.to(self.config['device'])
        self.policy.share_memory()
        if self.config['speaker']:
            for k in ['lambda', 'ent_target', 'ps_weight', 'ent_bonus', 'l_c', 'v_weight']:
                self.config[k] = self.config[k].to(self.config['device'])
        if self.config['listener']:
            for k in ['ce_weight', 'pl_weight', 'ent_bonus', 'l_c', 'v_weight']:
                self.config[k] = self.config[k].to(self.config['device'])
        if len(self.unique_agents) > 0:
            self.unique_agents = [s.to(self.config['device']) for s in self.unique_agents]

    def new_pairing(self, agent):
        """
        This function adds the new agents to the list of met agents
        Where the first if statement is to ensure agents are only added once
        This function is probably the best place for the copy heads functionality
        :param agent:
        :return:
        """
        # This next part is for the head copying functionality, if it is on. It copies after the 1st training loop
        if self.config['head_copy'] and len(self.unique_agents) == 1:  # 2nd condition only true after the first agent
            self.policy.copy_trained_head()
        if True not in [torch.all(i == agent.to(self.config['device'])) for i in self.unique_agents]:
            self.unique_agents.append(agent)
        self.unique_agents = [s.to(self.config['device']) for s in self.unique_agents]

    def forward(self, x, eps=0.0, train=False):
        log_prob, values = self.policy.forward(*x, index=self.calculate_index(x[-1]))
        p = Categorical(logits=log_prob)
        actions = p.sample()
        self.memory['log_prob'] = p.log_prob(actions)
        self.memory['action_dist'] = p
        if self.config['mode'] == 'q_critic':
            self.memory['values'] = values.gather(1, actions.view(-1, 1).long()).flatten()
        elif self.config['mode'] == 'baseline':
            self.memory['values'] = values.squeeze(1)
        else:
            self.memory['values'] = values
        self.memory['actions'] = [actions]
        if self.config['listener']:
            log_prob_no_m, _ = self.policy.forward(x[0], torch.zeros(x[1].shape).to(self.config['device']), x[2],
                                                   index=self.calculate_index(x[-1]))
            self.memory['action_dist_no_m'] = Categorical(logits=log_prob_no_m)
        if len(self.unique_agents) > 1:
            # line below removes the current agent from the update
            agents = [i for i in self.unique_agents if not torch.all(i == x[1])]
            log_prob_others = None
            if self.config['speaker_consistency']:
                log_prob_others, _ = self.policy.forward(x[0].repeat(len(agents), 1, 1, 1),
                                                         torch.cat(agents))
            if self.config['listener_consistency']:
                log_prob_others, _ = self.policy.forward(x[0].repeat(len(agents), 1, 1, 1),
                                                         torch.cat(agents),
                                                         x[2].repeat(len(agents), 1))

            if log_prob_others is not None:
                dist = Categorical(logits=log_prob_others)
                sampled = dist.sample()
                self.memory['log_prob_old_lang'] = dist.log_prob(sampled)
                self.memory['actions'].extend(sampled.chunk(len(agents)))
        return actions.detach(), self.memory['values'].detach()

    def add_to_buffer(self, x, r):
        self.memory['states'] = x
        self.memory['rewards'] = r

    def train(self):
        """
        :param batch_size:
        :param reset:
        :return:
        """
        loss, ent, ps, m_ent, m_cond_ent, pl, loss_no_m, c_loss = 0, None, None, None, None, None, None, None
        pi_loss, v_loss = None, None
        log_prob = self.memory['log_prob']
        returns = self.memory['rewards']
        if self.config['normalise_rewards']:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        if self.config['mode'] == 'q_critic':
            returns = self.memory['values'].detach()
            v_loss = self.config['v_weight'].squeeze(0) * F.mse_loss(self.memory['values'],
                                                                     self.memory['rewards'],
                                                                     reduction='sum')
            loss += v_loss
        elif self.config['mode'] == 'baseline':
            returns = returns - self.memory['values'].detach()
            v_loss = self.config['v_weight'].squeeze(0) * F.mse_loss(self.memory['values'],
                                                                     self.memory['rewards'],
                                                                     reduction='sum')
            loss += v_loss

        pi_loss = torch.sum(log_prob * returns)
        loss -= pi_loss

        if self.config['speaker']:
            ps = self.positive_signalling()
            loss += (self.config['ps_weight'] * ps).squeeze(0)
        if self.config['listener']:
            pl = self.positive_listening()
            loss += self.config['pl_weight'].squeeze(0) * pl
            ce = self.train_no_m()
            loss += self.config['ce_weight'].squeeze(0) * ce
        if (self.config['speaker_consistency'] or self.config['listener_consistency']) and len(self.unique_agents) > 1:
            # Now we can take the Q(s,a) from the image
            c_loss = self.memory['log_prob_old_lang'].sum() * self.config['l_c'].squeeze(0) # implicit R = 1
            loss -= c_loss #nll loss
        ent = self.memory['action_dist'].entropy().sum()
        loss -= self.config['ent_bonus'].squeeze(0) * ent
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.config['optimiser'].step()
        self.reset()
        self.iterations += 1
        return {'%s_total_loss' % self.config['name']: loss, '%s_ps' % self.config['name']: ps,
                '%s_m_ent' % self.config['name']: m_ent, '%s_m_cond_ent' % self.config['name']: m_cond_ent,
                '%s_pl' % self.config['name']: pl, '%s_loss_no_m' % self.config['name']: loss_no_m,
                '%s_ent' % self.config['name']: ent, '%s_consistency_loss' % self.config['name']: c_loss,
                '%s_pi_loss' % self.config['name']: pi_loss, '%s_v_loss' % self.config['name']: v_loss}

    def reset(self):
        self.config['optimiser'].zero_grad()
        self.memory = dict.fromkeys(['log_prob', 'rewards', 'action_dist', 'states', 'action_dist_no_m',
                                     'values', 'log_prob_old_lang', 'actions'])

    def mpol_ent(self):
        avg_p = self.memory['action_dist'].probs.mean(dim=0)
        return Categorical(avg_p).entropy()

    def mpol_condent(self):
        cond_ent = torch.pow(self.memory['action_dist'].entropy() - self.config['ent_target'], 2).sum()
        return cond_ent

    def positive_signalling(self):
        ps = -((self.config['batch_size']*self.mpol_ent()) - (self.config['lambda'] * self.mpol_condent()))
        return ps

    def positive_listening(self):
        """
        L1 norm between two policy distributions
        Want to maximise the distance between the two distributions
        :return: mean of the L1 norm
        """
        l1_norm = - torch.norm(self.memory['action_dist'].probs-self.memory['action_dist_no_m'].probs.detach(), p=1, dim=1)
        return l1_norm.sum()

    def train_no_m(self):
        """
        This section is for training the policy network which isn't conditioned on the message.
        Want to minimise this measure, Will be positive
        :return: loss for backpropagation
        """
        ce = -self.memory['action_dist'].probs.detach() * self.memory['action_dist_no_m'].logits
        return ce.sum(dim=1).sum()

    def save_model(self, index=0):
        if self.config['speaker']:
            type = 'speaker'
        else:
            type = 'listener'
        torch.save(self.policy.state_dict(), "trained_models/%s.pt"%(self.config['name']))

    def log_results(self, to_log):
        self.logger.log(to_log)

    def calculate_index(self, id):
        if self.config['feature_extraction']:
            id_current = int(torch.where(id[0] == 1)[0])
            my_id = int(self.config['name'][1])
            return (id_current - my_id) % 4
        else:
            return None




