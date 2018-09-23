# -*- encoding: utf-8 -*-
# stub: sass 3.6.0 ruby lib

Gem::Specification.new do |s|
  s.name = "sass"
  s.version = "3.6.0"

  s.required_rubygems_version = Gem::Requirement.new(">= 0") if s.respond_to? :required_rubygems_version=
  s.require_paths = ["lib"]
  s.authors = ["Natalie Weizenbaum", "Chris Eppstein", "Hampton Catlin"]
  s.date = "2018-09-19"
  s.description = "      Sass makes CSS fun again. Sass is an extension of CSS, adding\n      nested rules, variables, mixins, selector inheritance, and more.\n      It's translated to well-formatted, standard CSS using the\n      command line tool or a web-framework plugin.\n"
  s.email = "sass-lang@googlegroups.com"
  s.executables = ["sass", "sass-convert", "scss"]
  s.files = ["bin/sass", "bin/sass-convert", "bin/scss"]
  s.homepage = "http://sass-lang.com/"
  s.licenses = ["MIT"]
  s.post_install_message = "\nRuby Sass is deprecated and will be unmaintained as of 26 March 2019.\n\n* If you use Sass as a command-line tool, we recommend using Dart Sass, the new\n  primary implementation: https://sass-lang.com/install\n\n* If you use Sass as a plug-in for a Ruby web framework, we recommend using the\n  sassc gem: https://github.com/sass/sassc-ruby#readme\n\n* For more details, please refer to the Sass blog:\n  http://sass.logdown.com/posts/7081811\n\n"
  s.required_ruby_version = Gem::Requirement.new(">= 2.0.0")
  s.rubyforge_project = "sass"
  s.rubygems_version = "2.5.2.1"
  s.summary = "A powerful but elegant CSS compiler that makes CSS fun again."

  s.installed_by_version = "2.5.2.1" if s.respond_to? :installed_by_version

  if s.respond_to? :specification_version then
    s.specification_version = 4

    if Gem::Version.new(Gem::VERSION) >= Gem::Version.new('1.2.0') then
      s.add_runtime_dependency(%q<sass-listen>, ["~> 4.0.0"])
      s.add_development_dependency(%q<yard>, ["~> 0.8.7.6"])
      s.add_development_dependency(%q<redcarpet>, ["~> 3.3"])
      s.add_development_dependency(%q<nokogiri>, ["~> 1.6.0"])
      s.add_development_dependency(%q<minitest>, [">= 5"])
    else
      s.add_dependency(%q<sass-listen>, ["~> 4.0.0"])
      s.add_dependency(%q<yard>, ["~> 0.8.7.6"])
      s.add_dependency(%q<redcarpet>, ["~> 3.3"])
      s.add_dependency(%q<nokogiri>, ["~> 1.6.0"])
      s.add_dependency(%q<minitest>, [">= 5"])
    end
  else
    s.add_dependency(%q<sass-listen>, ["~> 4.0.0"])
    s.add_dependency(%q<yard>, ["~> 0.8.7.6"])
    s.add_dependency(%q<redcarpet>, ["~> 3.3"])
    s.add_dependency(%q<nokogiri>, ["~> 1.6.0"])
    s.add_dependency(%q<minitest>, [">= 5"])
  end
end
