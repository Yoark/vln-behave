class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env):
        self.env = env
        self.results = {}
        self.losses = [] # For learning agents

    def get_results(self, detailed_output=False):
        output = []
        for k, v in self.results.items():
            output.append({'instr_id': k, 'trajectory': v['path'], 'stop_prob': v['stop_prob'], 'final_logits':v['final_logits'],
            'candidates':v['candidates'], 'nav_type':v['nav_type'], 'prev_stop_probs': v['prev_stop_probs'], 'heading':v['heading'],
            'viewIndex': v['viewIndex'], 'next_pos':v['next_pos'], 'final_viewpoint_id': v['final_viewpoint_id'], 
            }) # added stop prob
            # import ipdb; ipdb.set_trace()
            if detailed_output:
                output[-1]['details'] = v['details']
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = traj
        else:   # Do a full round
            while True:
                for traj in self.rollout(**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = traj
                if looped:
                    break


