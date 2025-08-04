# Source Generated with Decompyle++
# File: train_options.cpython-37.pyc (Python 3.7)

from base_options import BaseOptions

class TrainOptions(BaseOptions):
    '''This class includes training options.

    It also includes shared options defined in BaseOptions.
    '''
    
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--display_freq', int, 400, 'frequency of showing training results on screen', **('type', 'default', 'help'))
        parser.add_argument('--display_ncols', int, 4, 'if positive, display all images in a single visdom web panel with certain number of images per row.', **('type', 'default', 'help'))
        parser.add_argument('--display_id', int, 1, 'window id of the web display', **('type', 'default', 'help'))
        parser.add_argument('--display_server', str, 'http://localhost', 'visdom server of the web display', **('type', 'default', 'help'))
        parser.add_argument('--display_env', str, 'main', 'visdom display environment name (default is "main")', **('type', 'default', 'help'))
        parser.add_argument('--display_port', int, 8097, 'visdom port of the web display', **('type', 'default', 'help'))
        parser.add_argument('--update_html_freq', int, 1000, 'frequency of saving training results to html', **('type', 'default', 'help'))
        parser.add_argument('--print_freq', int, 100, 'frequency of showing training results on console', **('type', 'default', 'help'))
        parser.add_argument('--no_html', 'store_true', 'do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/', **('action', 'help'))
        parser.add_argument('--save_latest_freq', int, 5000, 'frequency of saving the latest results', **('type', 'default', 'help'))
        parser.add_argument('--save_epoch_freq', int, 5, 'frequency of saving checkpoints at the end of epochs', **('type', 'default', 'help'))
        parser.add_argument('--save_by_iter', 'store_true', 'whether saves model by iteration', **('action', 'help'))
        parser.add_argument('--continue_train', 'store_true', 'continue training: load the latest model', **('action', 'help'))
        parser.add_argument('--epoch_count', int, 1, 'the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...', **('type', 'default', 'help'))
        parser.add_argument('--phase', str, 'train', 'train, val, test, etc', **('type', 'default', 'help'))
        parser.add_argument('--niter', int, 40, '# of iter at starting learning rate', **('type', 'default', 'help'))
        parser.add_argument('--niter_decay', int, 110, '# of iter to linearly decay learning rate to zero', **('type', 'default', 'help'))
        parser.add_argument('--beta1', float, 0.5, 'momentum term of adam', **('type', 'default', 'help'))
        parser.add_argument('--lr', float, 0.0002, 'initial learning rate for adam', **('type', 'default', 'help'))
        parser.add_argument('--gan_mode', str, 'lsgan', 'the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.', **('type', 'default', 'help'))
        parser.add_argument('--pool_size', int, 50, 'the size of image buffer that stores previously generated images', **('type', 'default', 'help'))
        parser.add_argument('--lr_policy', str, 'linear', 'learning rate policy. [linear | step | plateau | cosine]', **('type', 'default', 'help'))
        parser.add_argument('--lr_decay_iters', int, 50, 'multiply by a gamma every lr_decay_iters iterations', **('type', 'default', 'help'))
        self.isTrain = True
        return parser


